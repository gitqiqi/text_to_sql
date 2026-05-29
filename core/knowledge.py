# core/knowledge.py - KnowledgeBase：表结构查询、向量存取、格式化
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text

from config import get_database_config
from .db_manager import DatabasePoolManager
from .utils import _schema_cache, _vectors_cache, monitor_function


class KnowledgeBase:
    """负责表结构查询、向量存取、格式化（不再包含知识库/名词 CRUD，CRUD 见 repos.py）"""

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.engine = DatabasePoolManager.get_engine(db_name)

        self.cache_dir = Path("./cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dim = 384

    # ========== 向量操作方法 ==========

    def save_embeddings_to_holo(self, table_records: List[Dict], embeddings: np.ndarray):
        """批量保存向量到 Hologres（先删除再批量插入）"""
        if not table_records:
            return True

        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                delete_sql = "DELETE FROM knowledge.table_embeddings WHERE db_name = :db_name"
                conn.execute(text(delete_sql), {"db_name": self.db_name})

                insert_sql = """
                INSERT INTO knowledge.table_embeddings
                    (db_name, schema_name, table_name, table_comment, column_info,
                     vector_text, embedding, text_hash, schema_hash, updated_at)
                VALUES
                    (:db_name, :schema_name, :table_name, :table_comment, :column_info,
                     :vector_text, :embedding, :text_hash, :schema_hash, NOW())
                """

                batch_params = []
                for record, embedding in zip(table_records, embeddings):
                    vector_text = record.get('_vector_text', '')
                    text_hash = hashlib.md5(vector_text.encode()).hexdigest()
                    embedding_list = embedding.tolist()
                    column_info_json = json.dumps(record.get('_columns_json', []))

                    batch_params.append({
                        "db_name": self.db_name,
                        "schema_name": record.get('schema'),
                        "table_name": record['table_name'],
                        "table_comment": record.get('table_comment', ''),
                        "column_info": column_info_json,
                        "vector_text": vector_text,
                        "embedding": embedding_list,
                        "text_hash": text_hash,
                        "schema_hash": text_hash
                    })

                conn.execute(text(insert_sql), batch_params)
                trans.commit()
                _vectors_cache.invalidate(f"vectors:{self.db_name}")

                # 更新 schema fingerprint 元数据
                self._update_schema_fingerprint(conn, table_records)

                print(f"   ✅ 批量保存 {len(table_records)} 个向量到Hologres")
                return True

            except Exception as e:
                trans.rollback()
                print(f"   ❌ 保存向量失败: {e}")
                return False

    def _update_schema_fingerprint(self, conn, table_records: List[Dict]):
        """更新 schema fingerprint 到元数据表"""
        try:
            # 计算全库表结构的 fingerprint
            fingerprint = self._compute_schema_fingerprint(table_records)

            # 确保元数据表存在（DDL 单独事务）
            try:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS knowledge.db_metadata (
                        db_name VARCHAR(50) PRIMARY KEY,
                        schema_fingerprint VARCHAR(64) NOT NULL,
                        last_updated TIMESTAMP DEFAULT NOW()
                    )
                """))
                conn.commit()
            except Exception:
                # 表可能已存在，忽略
                pass

            # 插入或更新 fingerprint（DML 单独事务）
            conn.execute(text("""
                INSERT INTO knowledge.db_metadata (db_name, schema_fingerprint, last_updated)
                VALUES (:db_name, :fingerprint, NOW())
                ON CONFLICT (db_name) DO UPDATE SET
                    schema_fingerprint = EXCLUDED.schema_fingerprint,
                    last_updated = NOW()
            """), {"db_name": self.db_name, "fingerprint": fingerprint})
            conn.commit()
        except Exception as e:
            print(f"   ⚠️ 更新 schema fingerprint 失败: {e}")

    def _compute_schema_fingerprint(self, table_records: List[Dict]) -> str:
        """计算全库表结构的 fingerprint（所有表的 vector_text 拼接后 hash）"""
        # 按 schema.table 排序确保稳定性
        sorted_records = sorted(table_records, key=lambda r: f"{r.get('schema','')}.{r['table_name']}")
        combined = "\n".join(r.get('_vector_text', '') for r in sorted_records)
        return hashlib.md5(combined.encode()).hexdigest()

    def check_and_update_vectors(self):
        """检查表结构是否变化，变化则自动重建向量"""
        try:
            # 获取当前表结构
            table_records, vector_texts = self.get_vector_texts()
            if not table_records:
                print(f"   ⚠️ [{self.db_name}] 无表结构，跳过检查")
                return

            current_fingerprint = self._compute_schema_fingerprint(table_records)

            # 查询上次保存的 fingerprint
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT schema_fingerprint FROM knowledge.db_metadata
                    WHERE db_name = :db_name
                """), {"db_name": self.db_name})
                row = result.fetchone()

            if row is None:
                print(f"   🔄 [{self.db_name}] 首次建立向量库")
                self._rebuild_vectors(table_records, vector_texts)
            elif row[0] != current_fingerprint:
                print(f"   🔄 [{self.db_name}] 检测到表结构变化，重建向量")
                self._rebuild_vectors(table_records, vector_texts)
            else:
                print(f"   ✅ [{self.db_name}] 表结构无变化")

        except Exception as e:
            print(f"   ❌ [{self.db_name}] 检查向量失败: {e}")

    def _rebuild_vectors(self, table_records: List[Dict], vector_texts: List[str]):
        """重建向量（内部方法）"""
        from sentence_transformers import SentenceTransformer
        from .utils import SENTENCE_TRANSFORMER_MODEL

        print(f"   ├─ 加载向量模型...")
        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

        print(f"   ├─ 生成 {len(vector_texts)} 个向量...")
        batch_size = 50
        all_embeddings = []
        for i in range(0, len(vector_texts), batch_size):
            batch = vector_texts[i:i+batch_size]
            batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings)
        self.save_embeddings_to_holo(table_records, embeddings)
        print(f"   ✅ [{self.db_name}] 向量重建完成")

    # 类级别记录：已经验证过 SQL 路径不可用的 db 名（避免每次查询都先试 SQL 再 fallback）
    _sql_path_disabled = set()
    _sql_path_lock = threading.Lock()

    def vector_search_in_holo(self, query_embedding: List[float], top_k: int = 10,
                              schema_filter: Optional[str] = None) -> List[Dict]:
        """向量检索：优先走 Hologres Proxima 索引（快），失败时 fallback 到 Python numpy 计算。

        - 默认查 knowledge.table_embeddings_v2（带 Proxima 索引），用 pm_approx_squared_euclidean_distance
        - 如果函数/索引不可用（扩展没装），自动 fallback 到 numpy 内存计算
        - 用 _sql_path_disabled 记住失败过的 db，下次直接走 numpy
        - schema_filter: 逗号分隔的 schema 名，限定只在这些 schema 里检索
        """
        if self.db_name not in self._sql_path_disabled:
            try:
                results = self._vector_search_via_holo_sql(query_embedding, top_k, schema_filter)
                return results
            except Exception as e:
                with self._sql_path_lock:
                    self._sql_path_disabled.add(self.db_name)
                print(f"   ⚠️ Hologres Proxima 路径不可用（已切换到 numpy fallback）: {e}")

        return self._vector_search_via_numpy(query_embedding, top_k, schema_filter)

    def _vector_search_via_holo_sql(self, query_embedding: List[float], top_k: int,
                                    schema_filter: Optional[str] = None) -> List[Dict]:
        """使用 Hologres Proxima 索引查询 v2 表（需要 proxima 扩展）"""
        embedding_str = '{' + ','.join(str(x) for x in query_embedding) + '}'
        params = {"db_name": self.db_name, "query_embedding": embedding_str, "top_k": top_k}

        schema_clause = ""
        if schema_filter:
            schemas = [s.strip() for s in schema_filter.split(',') if s.strip()]
            if schemas:
                schema_clause = " AND schema_name = ANY(:schemas)"
                params["schemas"] = schemas

        sql = f"""
        SELECT
            schema_name,
            table_name,
            table_comment,
            column_info,
            vector_text,
            pm_approx_squared_euclidean_distance(embedding, CAST(:query_embedding AS float4[])) AS similarity_score
        FROM knowledge.table_embeddings
        WHERE db_name = :db_name{schema_clause}
        ORDER BY similarity_score ASC
        LIMIT :top_k
        """

        with self.engine.connect() as conn:
            try:
                conn.execute(text("SET hg_computing_resource = 'serverless'"))
            except Exception:
                pass

            result = conn.execute(text(sql), params)
            rows = result.fetchall()

        if not rows:
            return []

        results = []
        for i, row in enumerate(rows, 1):
            results.append({
                'schema': row[0],
                'table_name': row[1],
                'table_comment': row[2] or '',
                '_columns_json': row[3] if row[3] else [],
                '_vector_text': row[4] or '',
                '_similarity_score': float(row[5]),
                '_rank': i,
            })
        return results

    def _vector_search_via_numpy(self, query_embedding: List[float], top_k: int,
                                 schema_filter: Optional[str] = None) -> List[Dict]:
        """fallback：从 Hologres 拉出全部 embedding，在 Python 内存用 numpy 算 squared euclidean distance"""
        records, embeddings_matrix = self._load_all_vectors_cached()
        if not records:
            return []

        try:
            # schema 过滤：在距离计算前做，避免无关 schema 参与排序
            if schema_filter:
                schemas = set(s.strip().lower() for s in schema_filter.split(',') if s.strip())
                if schemas:
                    keep = [i for i, r in enumerate(records) if (r.get('schema') or '').lower() in schemas]
                    if not keep:
                        print(f"   ⚠️ schema 过滤后无可用向量（filter={schema_filter}）")
                        return []
                    embeddings_matrix = embeddings_matrix[keep]
                    records = [records[i] for i in keep]

            query_vec = np.array(query_embedding, dtype=np.float32)
            diff = embeddings_matrix - query_vec
            distances = np.einsum('ij,ij->i', diff, diff)

            n = len(distances)
            k = min(top_k, n)
            if k == n:
                top_indices = np.argsort(distances)
            else:
                cand = np.argpartition(distances, k)[:k]
                top_indices = cand[np.argsort(distances[cand])]

            results = []
            for rank, idx in enumerate(top_indices, 1):
                rec = records[idx]
                results.append({
                    'schema': rec['schema'],
                    'table_name': rec['table_name'],
                    'table_comment': rec['table_comment'],
                    '_columns_json': rec['column_info'],
                    '_vector_text': rec['vector_text'],
                    '_similarity_score': float(distances[idx]),
                    '_rank': rank,
                })
            return results
        except Exception as e:
            print(f"   ❌ numpy 向量检索失败: {e}")
            return []

    def _load_all_vectors_cached(self):
        """从 Hologres 拉取所有向量到内存，带 TTL 缓存"""
        cache_key = f"vectors:{self.db_name}"
        cached = _vectors_cache.get(cache_key)
        if cached is not None:
            return cached

        sql = """
        SELECT schema_name, table_name, table_comment, column_info, vector_text, embedding
        FROM knowledge.table_embeddings
        WHERE db_name = :db_name
        """

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), {"db_name": self.db_name})
                rows = result.fetchall()
        except Exception as e:
            print(f"   ❌ 加载向量失败: {e}")
            return [], np.array([])

        if not rows:
            return [], np.array([])

        records = []
        embedding_lists = []
        for row in rows:
            records.append({
                'schema': row[0],
                'table_name': row[1],
                'table_comment': row[2] or '',
                'column_info': row[3] if row[3] else [],
                'vector_text': row[4] or '',
            })
            embedding_lists.append(row[5])

        embeddings_matrix = np.array(embedding_lists, dtype=np.float32)
        result = (records, embeddings_matrix)
        _vectors_cache.set(cache_key, result)
        print(f"   ├─ 加载 {len(records)} 个表向量到内存（{embeddings_matrix.nbytes / 1024:.1f} KB）")
        return result

    def check_holo_vectors_exist(self) -> bool:
        sql = """
        SELECT COUNT(*) FROM knowledge.table_embeddings WHERE db_name = :db_name
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {"db_name": self.db_name})
            count = result.fetchone()[0]
            return count > 0

    def get_holo_vectors_count(self) -> int:
        sql = """
        SELECT COUNT(*) FROM knowledge.table_embeddings WHERE db_name = :db_name
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {"db_name": self.db_name})
            return result.fetchone()[0]

    # ========== 表结构获取方法 ==========

    @monitor_function
    def get_table_schema(self) -> Dict:
        cache_key = f"schema:{self.db_name}"
        cached = _schema_cache.get(cache_key)
        if cached is not None:
            print(f"    ├─ 命中表结构缓存: {self.db_name}")
            return cached

        print(f"    ├─ 加载表结构: {self.db_name}")

        db_config = get_database_config(self.db_name)
        if not db_config:
            return {'tables': {}}

        if db_config['type'] == 'postgresql':
            result = self._get_postgresql_schema()
        elif db_config['type'] == 'sqlite':
            result = self._get_sqlite_schema()
        else:
            result = {'tables': {}}

        _schema_cache.set(cache_key, result)
        return result

    def _get_postgresql_schema(self) -> Dict:
        print(f"    ├─ 获取PostgreSQL/Hologres表结构...")

        query = """
        WITH
        table_comments AS (
            SELECT
                n.nspname as schema_name,
                c.relname as table_name,
                COALESCE(pg_catalog.obj_description(c.oid, 'pg_class'), '') as table_comment
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind IN ('r', 'p', 'v', 'm')
                AND n.nspname NOT IN ('information_schema', 'pg_catalog', 'knowledge')
                AND n.nspname NOT LIKE 'pg_%%'
        ),
        column_comments AS (
            SELECT
                n.nspname as schema_name,
                c.relname as table_name,
                a.attname as column_name,
                a.attnum as column_order,
                COALESCE(pg_catalog.col_description(c.oid, a.attnum), '') as column_comment,
                pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE a.attnum > 0
                AND NOT a.attisdropped
                AND c.relkind IN ('r', 'p', 'v', 'm')
                AND n.nspname NOT IN ('information_schema', 'pg_catalog', 'knowledge')
                AND n.nspname NOT LIKE 'pg_%%'
        ),
        table_columns AS (
            SELECT
                schema_name,
                table_name,
                string_agg(
                    CASE
                        WHEN column_comment != '' THEN column_name || '[' || data_type || '](' || column_comment || ')'
                        ELSE column_name || '[' || data_type || ']'
                    END,
                    ', '
                    ORDER BY column_order
                ) as columns_text,
                string_agg(
                    CASE
                        WHEN column_comment != '' THEN '    ' || column_name || ' [' || data_type || '] (' || column_comment || ')'
                        ELSE '    ' || column_name || ' [' || data_type || ']'
                    END,
                    E'\n'
                    ORDER BY column_order
                ) as columns_formatted,
                string_agg(
                    CASE
                        WHEN column_comment != '' THEN column_comment
                        WHEN column_name ~ '[一-龥]' THEN column_name
                        ELSE NULL
                    END,
                    '、'
                    ORDER BY column_order
                ) as columns_for_vector,
                COUNT(*) as column_count,
                jsonb_agg(
                    jsonb_build_object('name', column_name, 'comment', column_comment, 'type', data_type)
                    ORDER BY column_order
                ) as columns_json
            FROM column_comments
            GROUP BY schema_name, table_name
        )
        SELECT
            tc.schema_name,
            tc.table_name,
            tc.table_comment,
            tcl.columns_text,
            tcl.columns_formatted,
            tcl.column_count,
            tcl.columns_json,
            TRIM(CONCAT(
                tc.schema_name, '.', tc.table_name,
                CASE WHEN tc.table_comment != '' THEN ' ' || tc.table_comment ELSE '' END,
                CASE WHEN tcl.columns_for_vector != '' THEN ' 字段: ' || tcl.columns_for_vector ELSE '' END
            )) as vector_text
        FROM table_comments tc
        JOIN table_columns tcl ON tc.schema_name = tcl.schema_name AND tc.table_name = tcl.table_name
        ORDER BY tc.schema_name, tc.table_name;
        """

        with self.engine.connect() as conn:
            try:
                conn.execute(text("ROLLBACK"))
            except Exception:
                pass

            result = conn.execute(text(query))
            rows = result.fetchall()

        print(f"    ├─ 获取到 {len(rows)} 个表")

        schemas = {}
        table_records = []
        all_vector_texts = []

        for row in rows:
            schema_name = row[0]
            table_name = row[1]
            table_comment = row[2]
            columns_text = row[3]
            columns_formatted = row[4]
            column_count = row[5]
            columns_json = row[6]
            vector_text = row[7]

            if schema_name not in schemas:
                schemas[schema_name] = {'tables': {}}

            schemas[schema_name]['tables'][table_name] = {
                '_table_comment': table_comment,
                '_columns_formatted': columns_formatted,
                '_column_count': column_count,
                '_vector_text': vector_text,
                '_columns_json': columns_json
            }

            col_names = [col.split('(')[0].strip() for col in columns_text.split(', ')] if columns_text else []
            table_records.append({
                'schema': schema_name,
                'table_name': table_name,
                'table_comment': table_comment,
                'columns': col_names,
                'column_count': column_count,
                '_vector_text': vector_text,
                '_columns_json': columns_json
            })
            all_vector_texts.append(vector_text)

        return {'schemas': schemas, 'table_records': table_records, 'vector_texts': all_vector_texts}

    def _get_sqlite_schema(self) -> Dict:
        print(f"    ├─ 获取SQLite表结构...")

        with self.engine.connect() as conn:
            tables_result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
            tables = {}
            for table_name in [row[0] for row in tables_result.fetchall()]:
                pragma_result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = {row[1]: row[2] for row in pragma_result.fetchall()}
                tables[table_name] = columns

        return {'tables': tables}

    def get_table_list(self) -> List[Dict]:
        schema_data = self.get_table_schema()
        table_list = []

        if 'schemas' in schema_data:
            for schema, schema_info in schema_data['schemas'].items():
                for table_name, table_info in schema_info.get('tables', {}).items():
                    table_list.append({
                        'schema': schema,
                        'table_name': table_name,
                        'table_comment': table_info.get('_table_comment', ''),
                        'column_count': table_info.get('_column_count', 0),
                        'columns': [col['name'] for col in table_info.get('_columns_json', [])]
                    })
        else:
            for table_name, table_info in schema_data.get('tables', {}).items():
                table_list.append({
                    'schema': None,
                    'table_name': table_name,
                    'table_comment': '',
                    'column_count': len(table_info),
                    'columns': list(table_info.keys())
                })

        return table_list

    def get_vector_texts(self) -> Tuple[List[Dict], List[str]]:
        schema_data = self.get_table_schema()
        if 'table_records' in schema_data and 'vector_texts' in schema_data:
            return schema_data['table_records'], schema_data['vector_texts']
        return [], []

    def get_table_schema_by_name(self, table_name: str) -> str:
        if not table_name:
            return ""

        schema_data = self.get_table_schema()

        if '.' in table_name:
            schema, table = table_name.split('.', 1)
        else:
            schema = None
            table = table_name

        if 'schemas' in schema_data:
            if schema and schema in schema_data['schemas']:
                table_info = schema_data['schemas'][schema]['tables'].get(table)
                if table_info:
                    columns_formatted = table_info.get('_columns_formatted')
                    table_comment = table_info.get('_table_comment', '')
                    header = f"表: {schema}.{table}"
                    if table_comment:
                        header += f" (注释: {table_comment})"
                    if columns_formatted:
                        return header + "\n列:\n" + columns_formatted
                    else:
                        return header

            for schema_name, schema_info in schema_data['schemas'].items():
                table_info = schema_info['tables'].get(table)
                if table_info:
                    columns_formatted = table_info.get('_columns_formatted')
                    table_comment = table_info.get('_table_comment', '')
                    header = f"表: {schema_name}.{table}"
                    if table_comment:
                        header += f" (注释: {table_comment})"
                    if columns_formatted:
                        return header + "\n列:\n" + columns_formatted
                    else:
                        return header

        return f"未找到表: {table_name}"

    def get_formatted_schema(self, selected_tables: Optional[List[str]] = None, schema_filter: Optional[str] = None) -> str:
        schema_data = self.get_table_schema()

        if selected_tables:
            return self._format_selected_tables(schema_data, selected_tables)
        else:
            return self._format_all_tables(schema_data, schema_filter=schema_filter)

    def _format_all_tables(self, schema_data: Dict, schema_filter: Optional[str] = None) -> str:
        formatted = []
        filter_set = None
        if schema_filter:
            filter_set = set(s.strip() for s in schema_filter.split(',') if s.strip())

        if 'schemas' in schema_data:
            for schema_name, schema_info in schema_data['schemas'].items():
                if filter_set and schema_name not in filter_set:
                    continue
                for table_name, table_info in schema_info.get('tables', {}).items():
                    columns_formatted = table_info.get('_columns_formatted')
                    table_comment = table_info.get('_table_comment', '')

                    header = f"表名: {schema_name}.{table_name}"
                    if table_comment:
                        header += f" 表注释: {table_comment}"

                    if columns_formatted:
                        formatted.append(header + "\n列:\n" + columns_formatted)
                    else:
                        formatted.append(header)

        return "\n\n".join(formatted) if formatted else "无可用表结构"

    def _format_selected_tables(self, schema_data: Dict, selected_tables: List[str]) -> str:
        formatted = []
        found_count = 0

        for full_name in selected_tables:
            if '.' in full_name:
                schema_name, table_name = full_name.split('.', 1)
            else:
                schema_name = None
                table_name = full_name

            table_info = None

            if 'schemas' in schema_data:
                if schema_name and schema_name in schema_data['schemas']:
                    table_info = schema_data['schemas'][schema_name]['tables'].get(table_name)
                else:
                    for s_name, s_info in schema_data['schemas'].items():
                        if table_name in s_info['tables']:
                            table_info = s_info['tables'][table_name]
                            schema_name = s_name
                            break

            if table_info:
                columns_formatted = table_info.get('_columns_formatted')
                table_comment = table_info.get('_table_comment', '')
                header = f"表名: {schema_name}.{table_name}"
                if table_comment:
                    header += f" 表注释: {table_comment}"
                if columns_formatted:
                    formatted.append(header + "\n列:\n" + columns_formatted)
                else:
                    formatted.append(header)
                found_count += 1
            else:
                print(f"   ⚠️ 未找到表: {full_name}")

        if not formatted:
            return f"未找到指定的表: {', '.join(selected_tables)}"

        print(f"   ✅ 成功格式化 {found_count}/{len(selected_tables)} 个表")
        return "\n\n".join(formatted)


# ========== 后台监控线程 ==========

_monitor_thread = None
_monitor_lock = threading.Lock()

def start_vector_monitor(db_names: List[str], check_interval: int = 600):
    """启动后台监控线程，定期检查表结构变化并自动更新向量

    Args:
        db_names: 要监控的数据库名称列表
        check_interval: 检查间隔（秒），默认 600 秒（10 分钟）
    """
    global _monitor_thread

    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            print("   ⚠️ 向量监控线程已在运行")
            return

        def monitor_loop():
            print(f"🔍 向量监控线程启动，每 {check_interval // 60} 分钟检查一次")

            # 启动时立即检查一次
            for db_name in db_names:
                try:
                    kb = KnowledgeBase(db_name)
                    kb.check_and_update_vectors()
                except Exception as e:
                    print(f"   ❌ [{db_name}] 初始检查失败: {e}")

            # 定时循环检查
            while True:
                time.sleep(check_interval)
                print(f"\n🔍 [{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始定期检查...")
                for db_name in db_names:
                    try:
                        kb = KnowledgeBase(db_name)
                        kb.check_and_update_vectors()
                    except Exception as e:
                        print(f"   ❌ [{db_name}] 检查失败: {e}")

        _monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name="VectorMonitor")
        _monitor_thread.start()
        print(f"   ✅ 向量监控线程已启动（监控 {len(db_names)} 个数据库）")

