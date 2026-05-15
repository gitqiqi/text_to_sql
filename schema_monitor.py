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
from app_core import KnowledgeBase, DatabasePoolManager
from config import get_database_config, get_available_databases

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
        """确保 table_embeddings 表有 schema_hash 字段"""
        try:
            check_sql = """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_schema = 'knowledge' 
                  AND table_name = 'table_embeddings' 
                  AND column_name = 'schema_hash'
            )
            """
            with self.engine.connect() as conn:
                result = conn.execute(text(check_sql))
                exists = result.fetchone()[0]
                
                if not exists:
                    # 先确保表存在
                    create_table_sql = """
                    CREATE SCHEMA IF NOT EXISTS knowledge;
                    CREATE TABLE IF NOT EXISTS knowledge.table_embeddings (
                        id SERIAL PRIMARY KEY,
                        db_name VARCHAR(50) NOT NULL,
                        schema_name VARCHAR(100),
                        table_name VARCHAR(100) NOT NULL,
                        table_comment TEXT,
                        column_info JSONB,
                        vector_text TEXT,
                        embedding float4[],
                        text_hash VARCHAR(64),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                    """
                    conn.execute(text(create_table_sql))
                    
                    # 添加 schema_hash 列
                    alter_sql = """
                    ALTER TABLE knowledge.table_embeddings 
                    ADD COLUMN schema_hash VARCHAR(64)
                    """
                    conn.execute(text(alter_sql))
                    
                    # 创建索引
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_table_embeddings_db_name ON knowledge.table_embeddings(db_name)"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_table_embeddings_text_hash ON knowledge.table_embeddings(text_hash)"))
                    
                    conn.commit()
                    logger.info(f"已创建向量表并添加 schema_hash 列")
        except Exception as e:
            logger.warning(f"初始化向量表时出现问题: {e}")
    
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
    """向量增量更新器（带模型缓存）"""
    
    # 类级别的模型缓存
    _model = None
    _model_lock = threading.Lock()
    
    @classmethod
    def _get_model(cls):
        """获取缓存的 SentenceTransformer 模型"""
        if cls._model is None:
            with cls._model_lock:
                if cls._model is None:
                    from sentence_transformers import SentenceTransformer
                    logger.info("🔄 首次加载向量模型...")
                    load_start = time.time()
                    cls._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                    load_duration = (time.time() - load_start) * 1000
                    logger.info(f"✅ 模型加载完成，耗时: {load_duration:.2f} ms")
        return cls._model
    
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.engine = DatabasePoolManager.get_engine(db_name)
        self.kb = KnowledgeBase(db_name)
    
    def update_single_table(self, schema: str, table_name: str, 
                            vector_text: str, columns_json: List) -> bool:
        """更新单个表的向量"""
        try:
            import numpy as np
            
            logger.info(f"更新表向量: {schema}.{table_name}")
            
            model = self._get_model()
            embedding = model.encode([vector_text], convert_to_numpy=True)[0]
            
            text_hash = hashlib.md5(vector_text.encode()).hexdigest()
            embedding_list = embedding.tolist()
            schema_hash = text_hash
            
            delete_sql = """
            DELETE FROM knowledge.table_embeddings 
            WHERE db_name = :db_name 
              AND schema_name = :schema_name 
              AND table_name = :table_name
            """
            
            insert_sql = """
            INSERT INTO knowledge.table_embeddings 
                (db_name, schema_name, table_name, table_comment, column_info, 
                 vector_text, embedding, text_hash, schema_hash, updated_at)
            VALUES 
                (:db_name, :schema_name, :table_name, :table_comment, :column_info, 
                 :vector_text, :embedding, :text_hash, :schema_hash, NOW())
            """
            
            with self.engine.connect() as conn:
                conn.execute(
                    text(delete_sql),
                    {"db_name": self.db_name, "schema_name": schema, "table_name": table_name}
                )
                
                conn.execute(
                    text(insert_sql),
                    {
                        "db_name": self.db_name,
                        "schema_name": schema,
                        "table_name": table_name,
                        "table_comment": "",
                        "column_info": json.dumps(columns_json),
                        "vector_text": vector_text,
                        "embedding": embedding_list,
                        "text_hash": text_hash,
                        "schema_hash": schema_hash
                    }
                )
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
        from app_core import precompute_all_embeddings
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
        """检查并更新指定数据库"""
        try:
            comparator = self.comparators.get(db_name)
            updater = self.updaters.get(db_name)
            
            if not comparator or not updater:
                return
            
            current = comparator.get_current_schema()
            stored = comparator.get_stored_schema_hash()
            
            # 首次运行，全量构建
            if not stored or len(stored) == 0:
                logger.info(f"数据库 {db_name} 首次监控，正在构建向量...")
                updater.rebuild_all()
                comparator.save_schema_hash(current)
                return
            
            changes = comparator.find_changes(current, stored)
            total_changes = len(changes['added']) + len(changes['deleted']) + len(changes['modified'])
            
            if total_changes > 0:
                logger.info(f"数据库 {db_name} 发现 {total_changes} 个变化: "
                          f"+{len(changes['added'])} -{len(changes['deleted'])} ~{len(changes['modified'])}")
                
                for key in changes['deleted']:
                    if not self.running:
                        return
                    parts = key.split('.')
                    schema = parts[0] if len(parts) > 1 else 'public'
                    table_name = parts[-1]
                    updater.delete_single_table(schema, table_name)
                
                for key in changes['added'] + changes['modified']:
                    if not self.running:
                        return
                    info = current[key]
                    updater.update_single_table(
                        info['schema'], info['table_name'],
                        info['vector_text'], info['columns_json']
                    )
                
                comparator.save_schema_hash(current)
                logger.info(f"数据库 {db_name} 同步完成")
            else:
                logger.debug(f"数据库 {db_name} 无变化")
            
        except Exception as e:
            logger.error(f"检查数据库 {db_name} 时出错: {e}")
    
    def _signal_handler(self, signum, frame):
        logger.info(f"收到停止信号，正在优雅关闭...")
        self.running = False


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
        db_names = [config['id'] for config in db_configs]
    
    if not db_names:
        logger.error("没有找到可监控的数据库")
        sys.exit(1)
    
    logger.info(f"目标数据库: {', '.join(db_names)}")
    
    # 重建模式
    if args.rebuild:
        from app_core import precompute_all_embeddings
        for db_name in db_names:
            logger.info(f"重建数据库 {db_name} 的向量...")
            precompute_all_embeddings(db_name, force_rebuild=True)
        return
    
    # 监控模式
    service = SchemaMonitorService(db_names, args.interval)
    
    try:
        if args.once:
            for db_name in db_names:
                service._check_and_update(db_name)
        else:
            service.start()
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在退出...")
        service.running = False
        sys.exit(0)


if __name__ == '__main__':
    main()