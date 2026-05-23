# core/knowledge.py - KnowledgeBase：表结构查询、向量存取、格式化
import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text

from config import get_database_config
from .db_manager import DatabasePoolManager
from .utils import _schema_cache, monitor_function


class KnowledgeBase:
    """负责表结构查询、向量存取、格式化（不再包含知识库/名词 CRUD，CRUD 见 repos.py）"""

    _table_initialized = set()
    _init_lock = threading.Lock()

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.engine = DatabasePoolManager.get_engine(db_name)

        if db_name not in self._table_initialized:
            with self._init_lock:
                if db_name not in self._table_initialized:
                    self._ensure_vector_table()
                    self._table_initialized.add(db_name)

        self.cache_dir = Path("./cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dim = 384

    def _ensure_vector_table(self):
        """确保向量表存在"""
        create_table_sql = """
        CREATE SCHEMA IF NOT EXISTS knowledge;
        CREATE TABLE IF NOT EXISTS knowledge.table_embeddings (
            id BIGSERIAL PRIMARY KEY,
            db_name VARCHAR(50) NOT NULL,
            schema_name VARCHAR(100),
            table_name VARCHAR(100) NOT NULL,
            table_comment TEXT,
            column_info JSONB,
            vector_text TEXT,
            embedding FLOAT4[] CHECK(array_ndims(embedding) = 1 AND array_length(embedding, 1) = 384),
            text_hash VARCHAR(64),
            schema_hash VARCHAR(64),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
                print(f"   ✅ 向量表已就绪")
        except Exception as e:
            print(f"   ⚠️ 向量表创建失败: {e}")

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
                print(f"   ✅ 批量保存 {len(table_records)} 个向量到Hologres")
                return True

            except Exception as e:
                trans.rollback()
                print(f"   ❌ 保存向量失败: {e}")
                return False

    def vector_search_in_holo(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """在Hologres中进行向量检索"""

        embedding_str = '{' + ','.join(str(x) for x in query_embedding) + '}'

        sql = """
        SELECT
            schema_name,
            table_name,
            table_comment,
            column_info,
            vector_text,
            pm_approx_squared_euclidean_distance(embedding, CAST(:query_embedding AS float4[])) AS similarity_score
        FROM knowledge.table_embeddings
        WHERE db_name = :db_name
        ORDER BY similarity_score ASC
        LIMIT :top_k
        """

        try:
            with self.engine.connect() as conn:
                try:
                    conn.execute(text("SET hg_computing_resource = 'serverless'"))
                except Exception:
                    pass

                result = conn.execute(
                    text(sql),
                    {
                        "db_name": self.db_name,
                        "query_embedding": embedding_str,
                        "top_k": top_k
                    }
                )

                rows = result.fetchall()

                if not rows:
                    return []

                results = []
                for i, row in enumerate(rows, 1):
                    columns_json = row[3] if row[3] else []

                    results.append({
                        'schema': row[0],
                        'table_name': row[1],
                        'table_comment': row[2] or '',
                        '_columns_json': columns_json,
                        '_vector_text': row[4] or '',
                        '_similarity_score': float(row[5]),
                        '_rank': i
                    })

                return results

        except Exception as e:
            print(f"   ❌ Hologres向量检索失败: {e}")
            return []

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
                '表名: ', tc.schema_name, '.', tc.table_name,
                CASE WHEN tc.table_comment != '' THEN ' 表注释: ' || tc.table_comment ELSE '' END,
                ' 列: ', tcl.columns_text
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
