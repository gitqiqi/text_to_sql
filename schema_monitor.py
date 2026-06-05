#!/usr/bin/env python
# schema_monitor.py - 表结构监控与向量增量更新服务

import os
import sys
import time
import json
import hashlib
import logging
import signal
import argparse
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from core import KnowledgeBase, DatabasePoolManager, EMBEDDING_DIM
from core.embedding_client import get_embedding_model
from config import get_database_config, get_available_databases, EXCLUDED_DATABASES

load_dotenv()

# ==================== 配置 ====================
POLL_INTERVAL = int(os.getenv('SCHEMA_POLL_INTERVAL', '60'))
ENABLE_AUTO_UPDATE = os.getenv('ENABLE_AUTO_UPDATE', 'true').lower() == 'true'

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== 表结构对比器 ====================
class SchemaComparator:
    """表结构对比器（直接使用 table_embeddings 表）"""
    
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.kb = KnowledgeBase(db_name)
        self.engine = DatabasePoolManager.get_engine(db_name)
        self._ensure_hash_column()
    
    def _ensure_hash_column(self):
        """确保 schema_hash 列存在（表需提前通过 sql/create_table_embeddings.sql 创建）"""
        # 表由 DBA 提前建好，此处仅做兼容性检查
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = 'knowledge'
                          AND table_name = 'table_embeddings'
                          AND column_name = 'schema_hash'
                    )
                """))
                if not result.fetchone()[0]:
                    logger.warning("table_embeddings 缺少 schema_hash 列，请执行 sql/create_table_embeddings.sql")
        except Exception as e:
            logger.warning(f"检查 table_embeddings 时出现问题: {e}")
    
    def get_current_signatures(self) -> Dict[str, str]:
        """轻量获取当前 schema 签名（key -> md5 hash），不取完整字段信息

        签名计算方式必须和 _get_postgresql_schema 中 vector_text 的格式完全一致：
            '表名: schema.table' [+ ' 表注释: comment'] + ' 列: col1[type](comment), col2[type], ...'
        这样跟 knowledge.table_embeddings.schema_hash 可以直接比对。
        """
        sql = """
        WITH column_data AS (
            SELECT
                n.nspname as schema_name,
                c.relname as table_name,
                a.attname as col_name,
                a.attnum as col_order,
                COALESCE(pg_catalog.col_description(c.oid, a.attnum), '') as col_comment,
                pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type,
                COALESCE(pg_catalog.obj_description(c.oid, 'pg_class'), '') as table_comment
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE a.attnum > 0
                AND NOT a.attisdropped
                AND c.relkind IN ('r', 'p', 'v', 'm')
                AND n.nspname NOT IN ('information_schema', 'pg_catalog', 'knowledge')
                and c.relname not like '%%middle%%'
        )
        SELECT
            schema_name,
            table_name,
            MD5(
                TRIM(
                    '表名: ' || schema_name || '.' || table_name ||
                    CASE WHEN MAX(table_comment) != '' THEN ' 表注释: ' || MAX(table_comment) ELSE '' END ||
                    ' 列: ' || string_agg(
                        CASE
                            WHEN col_comment != '' THEN col_name || '[' || data_type || '](' || col_comment || ')'
                            ELSE col_name || '[' || data_type || ']'
                        END,
                        ', '
                        ORDER BY col_order
                    )
                )
            ) as sig
        FROM column_data
        GROUP BY schema_name, table_name
        """

        signatures = {}
        with self.engine.connect() as conn:
            try:
                conn.execute(text("ROLLBACK"))
            except Exception:
                pass
            try:
                conn.execute(text("SET hg_computing_resource = 'serverless'"))
            except Exception:
                pass

            result = conn.execute(text(sql))
            for row in result:
                schema = row[0] or 'public'
                table_name = row[1]
                sig = row[2]
                key = f"{schema}.{table_name}"
                signatures[key] = sig

        return signatures

    def get_current_schema(self) -> Dict[str, Dict]:
        """获取当前数据库的表结构及哈希"""
        table_records, vector_texts = self.kb.get_vector_texts()

        current_schema = {}
        for record, vector_text in zip(table_records, vector_texts):
            schema = record.get('schema', 'public')
            table_name = record['table_name']
            key = f"{schema}.{table_name}"
            table_hash = hashlib.md5(vector_text.encode()).hexdigest()

            current_schema[key] = {
                'hash': table_hash,
                'schema': schema,
                'table_name': table_name,
                'table_comment': record.get('table_comment', ''),
                'vector_text': vector_text,
                'columns_json': record.get('_columns_json', [])
            }
        
        return current_schema
    
    def get_stored_schema_hash(self) -> Dict[str, str]:
        """从 table_embeddings 表中获取存储的表结构哈希"""
        sql = """
        SELECT schema_name, table_name, schema_hash
        FROM knowledge.table_embeddings
        WHERE db_name = :db_name
        """

        stored_hash = {}
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {"db_name": self.db_name})
            for row in result:
                schema = row[0] or 'public'
                table_name = row[1]
                key = f"{schema}.{table_name}"
                stored_hash[key] = row[2] if row[2] else ''

        return stored_hash
    
    def save_schema_hash(self, schema_info: Dict):
        """保存表结构哈希到 table_embeddings 表"""
        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                update_sql = """
                UPDATE knowledge.table_embeddings
                SET schema_hash = :schema_hash, updated_at = NOW()
                WHERE db_name = :db_name
                  AND schema_name = :schema_name
                  AND table_name = :table_name
                """

                for key, info in schema_info.items():
                    conn.execute(
                        text(update_sql),
                        {
                            "db_name": self.db_name,
                            "schema_name": info['schema'],
                            "table_name": info['table_name'],
                            "schema_hash": info['hash']
                        }
                    )

                trans.commit()
                logger.debug(f"已保存 {len(schema_info)} 个表的哈希值")
            except Exception as e:
                trans.rollback()
                logger.error(f"保存哈希失败: {e}")
    
    def find_changes(self, current: Dict, stored: Dict) -> Dict:
        """对比并找出变化"""
        current_keys = set(current.keys())
        stored_keys = set(stored.keys())
        
        return {
            'added': list(current_keys - stored_keys),
            'deleted': list(stored_keys - current_keys),
            'modified': [k for k in current_keys & stored_keys 
                        if current[k]['hash'] != stored[k]]
        }


# ==================== 向量更新器（带模型缓存） ====================
class VectorUpdater:
    """向量增量更新器（带模型缓存，支持双列）"""
    
    # 类级别的模型缓存（per-provider）
    _models = {}
    _model_lock = threading.Lock()

    @classmethod
    def _get_model(cls, provider: str = None):
        """获取缓存的向量模型"""
        if provider not in cls._models:
            with cls._model_lock:
                if provider not in cls._models:
                    logger.info(f"🔄 首次加载向量模型 ({provider or 'default'})...")
                    load_start = time.time()
                    cls._models[provider] = get_embedding_model(provider)
                    load_duration = (time.time() - load_start) * 1000
                    logger.info(f"✅ 模型加载完成，耗时: {load_duration:.2f} ms")
        return cls._models[provider]

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.engine = DatabasePoolManager.get_engine(db_name)
        self.kb = KnowledgeBase(db_name)
    
    def update_single_table(self, schema: str, table_name: str, 
                            vector_text: str, columns_json: List) -> bool:
        """更新单个表的向量（两个模型列都更新）"""
        try:
            import numpy as np
            
            logger.info(f"更新表向量: {schema}.{table_name}")
            
            for provider in ('local', 'api'):
                model = self._get_model(provider)
                embedding = model.encode([vector_text], convert_to_numpy=True)[0]
                
                text_hash = hashlib.md5(vector_text.encode()).hexdigest()
                embedding_list = embedding.tolist()
                
                embedding_col = KnowledgeBase._embedding_col(provider)
                
                update_sql = f"""
                UPDATE knowledge.table_embeddings
                SET {embedding_col} = :embedding,
                    vector_text = :vector_text,
                    text_hash = :text_hash,
                    schema_hash = :schema_hash,
                    table_comment = :table_comment,
                    column_info = CAST(:column_info AS jsonb),
                    updated_at = NOW()
                WHERE db_name = :db_name
                  AND schema_name = :schema_name
                  AND table_name = :table_name
                """

                insert_sql = f"""
                INSERT INTO knowledge.table_embeddings
                    (db_name, schema_name, table_name, table_comment, column_info,
                     vector_text, {embedding_col}, text_hash, schema_hash, updated_at)
                VALUES
                    (:db_name, :schema_name, :table_name, :table_comment, CAST(:column_info AS jsonb),
                     :vector_text, :embedding, :text_hash, :schema_hash, NOW())
                """

                params = {
                    "db_name": self.db_name,
                    "schema_name": schema,
                    "table_name": table_name,
                    "table_comment": "",
                    "column_info": json.dumps(columns_json),
                    "vector_text": vector_text,
                    "embedding": embedding_list,
                    "text_hash": text_hash,
                    "schema_hash": text_hash,
                }

                with self.engine.connect() as conn:
                    result = conn.execute(text(update_sql), params)
                    if result.rowcount == 0:
                        conn.execute(text(insert_sql), params)
                    conn.commit()

            logger.info(f"✅ 更新成功: {schema}.{table_name}")
            return True

        except Exception as e:
            logger.error(f"更新失败 {schema}.{table_name}: {e}")
            return False
    
    def delete_single_table(self, schema: str, table_name: str) -> bool:
        """删除表的向量"""
        sql = """
        DELETE FROM knowledge.table_embeddings 
        WHERE db_name = :db_name 
          AND schema_name = :schema_name 
          AND table_name = :table_name
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(sql),
                    {"db_name": self.db_name, "schema_name": schema, "table_name": table_name}
                )
                conn.commit()
                
                if result.rowcount > 0:
                    logger.info(f"✅ 删除成功: {schema}.{table_name}")
                return True
        except Exception as e:
            logger.error(f"删除失败 {schema}.{table_name}: {e}")
            return False
    
    def rebuild_all(self) -> bool:
        """全量重建"""
        from core import precompute_all_embeddings
        logger.info(f"开始全量重建 {self.db_name} 的向量...")
        precompute_all_embeddings(self.db_name, force_rebuild=True)
        logger.info(f"全量重建完成")
        return True


# ==================== 监控服务 ====================
class SchemaMonitorService:
    """表结构监控服务"""
    
    def __init__(self, db_names: List[str], poll_interval: int = POLL_INTERVAL):
        self.poll_interval = poll_interval
        self.running = False
        self._stop_count = 0
        self.comparators: Dict[str, SchemaComparator] = {}
        self.updaters: Dict[str, VectorUpdater] = {}
        self.db_names = []
        
        for db_name in db_names:
            try:
                config = get_database_config(db_name)
                if not config:
                    logger.warning(f"跳过数据库 {db_name}：配置不存在")
                    continue
                
                if config.get('type') == 'postgresql':
                    host = config.get('host')
                    if not host or host == 'None' or host == '':
                        logger.warning(f"跳过数据库 {db_name}：host 配置无效")
                        continue
                
                self.comparators[db_name] = SchemaComparator(db_name)
                self.updaters[db_name] = VectorUpdater(db_name)
                self.db_names.append(db_name)
                logger.info(f"✅ 初始化数据库: {db_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ 跳过数据库 {db_name}：{e}")
        
        if not self.db_names:
            logger.error("没有可监控的数据库")
    
    def start(self):
        """启动监控服务"""
        if not self.db_names:
            logger.error("无有效数据库，监控服务退出")
            return
        
        self.running = True
        logger.info(f"监控服务启动，轮询间隔: {self.poll_interval}秒")
        logger.info(f"监控数据库: {', '.join(self.db_names)}")
        logger.info("按 Ctrl+C 停止服务")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 首次运行立即检查
        for db_name in self.db_names:
            if not self.running:
                break
            self._check_and_update(db_name)
        
        # 主循环
        while self.running:
            try:
                for db_name in self.db_names:
                    if not self.running:
                        break
                    self._check_and_update(db_name)
                
                # 分段 sleep，便于及时响应停止信号
                for _ in range(self.poll_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                if self.running:
                    time.sleep(5)
    
    def _check_and_update(self, db_name: str):
        """检查并更新指定数据库（先用轻量签名对比，发现变化才拉完整 schema）"""
        try:
            comparator = self.comparators.get(db_name)
            updater = self.updaters.get(db_name)

            if not comparator or not updater:
                return

            t0 = time.time()
            current_sigs = comparator.get_current_signatures()
            stored = comparator.get_stored_schema_hash()
            sig_duration = (time.time() - t0) * 1000
            logger.debug(f"[{db_name}] 轻量签名对比耗时: {sig_duration:.0f} ms，{len(current_sigs)} 个表")

            # 首次运行（DB 完全空），全量构建
            if not stored or len(stored) == 0:
                logger.info(f"数据库 {db_name} 首次监控，正在构建向量...")
                updater.rebuild_all()
                # 重建后再读 hash 保存（rebuild_all 已经写入了 schema_hash，无需重复）
                return

            current_keys = set(current_sigs.keys())
            stored_keys = set(stored.keys())

            added_keys = list(current_keys - stored_keys)
            deleted_keys = list(stored_keys - current_keys)
            modified_keys = [k for k in current_keys & stored_keys if current_sigs[k] != stored[k]]

            total_changes = len(added_keys) + len(deleted_keys) + len(modified_keys)
            if total_changes == 0:
                logger.debug(f"数据库 {db_name} 无变化")
                return

            logger.info(f"数据库 {db_name} 发现 {total_changes} 个变化: "
                        f"+{len(added_keys)} -{len(deleted_keys)} ~{len(modified_keys)}")

            # 删除的表：直接删，无需拉完整 schema
            for key in deleted_keys:
                if not self.running:
                    return
                parts = key.split('.')
                schema = parts[0] if len(parts) > 1 else 'public'
                table_name = parts[-1]
                updater.delete_single_table(schema, table_name)

            # 仅当有 added 或 modified 时，才拉完整 schema 拿 vector_text 和字段详情
            keys_to_update = added_keys + modified_keys
            if keys_to_update:
                current = comparator.get_current_schema()
                for key in keys_to_update:
                    if not self.running:
                        return
                    info = current.get(key)
                    if not info:
                        logger.warning(f"未在完整 schema 中找到 {key}，跳过")
                        continue
                    updater.update_single_table(
                        info['schema'], info['table_name'],
                        info['vector_text'], info['columns_json']
                    )

            logger.info(f"数据库 {db_name} 同步完成")

        except Exception as e:
            logger.error(f"检查数据库 {db_name} 时出错: {e}", exc_info=True)
    
    def _signal_handler(self, signum, frame):
        self._stop_count += 1
        if self._stop_count == 1:
            logger.info("收到停止信号，正在优雅关闭...（再按一次 Ctrl+C 强制退出）")
            self.running = False
        else:
            logger.warning("再次收到停止信号，强制退出")
            os._exit(1)


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='表结构监控服务')
    parser.add_argument('--interval', type=int, default=POLL_INTERVAL,
                        help=f'检查间隔（秒），默认: {POLL_INTERVAL}')
    parser.add_argument('--db', nargs='+', help='指定监控的数据库')
    parser.add_argument('--once', action='store_true', help='只运行一次，不循环')
    parser.add_argument('--rebuild', action='store_true', help='强制全量重建所有向量')
    
    args = parser.parse_args()
    
    # 确定监控的数据库
    if args.db:
        db_names = args.db
    else:
        db_configs = get_available_databases()
        db_names = [c['id'] for c in db_configs if c['id'] not in EXCLUDED_DATABASES]
    
    if not db_names:
        logger.error("没有找到可监控的数据库")
        sys.exit(1)
    
    logger.info(f"目标数据库: {', '.join(db_names)}")
    
    # 重建模式
    if args.rebuild:
        from core import precompute_all_embeddings
        for db_name in db_names:
            logger.info(f"重建数据库 {db_name} 的向量...")
            precompute_all_embeddings(db_name, force_rebuild=True)
        return
    
    # 监控模式
    service = SchemaMonitorService(db_names, args.interval)
    
    try:
        if args.once:
            for db_name in db_names:
                if db_name in EXCLUDED_DATABASES:
                    continue
                service._check_and_update(db_name)
        else:
            service.start()
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在退出...")
        service.running = False
        sys.exit(0)


if __name__ == '__main__':
    main()