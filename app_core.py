# app_core.py - 核心类和公共函数（Hologres 适配版）
from pathlib import Path
import time
import os
import threading
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
from functools import wraps
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
from volcenginesdkarkruntime import Ark
from sentence_transformers import SentenceTransformer
import numpy as np
import json

load_dotenv()

from config import (
    get_database_config,
    get_available_databases
)

# ==================== 配置 ====================
POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '5'))
MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '10'))
POOL_PRE_PING = os.getenv('DB_POOL_PRE_PING', 'true').lower() == 'true'

# 向量模型路径配置
_model_path_config = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')

# 解析路径
if _model_path_config.startswith('../') or _model_path_config.startswith('./'):
    current_file_dir = Path(__file__).parent.absolute()
    SENTENCE_TRANSFORMER_MODEL = str(current_file_dir / _model_path_config)
elif not _model_path_config.startswith('/') and not _model_path_config.startswith('http'):
    current_file_dir = Path(__file__).parent.absolute()
    SENTENCE_TRANSFORMER_MODEL = str(current_file_dir / _model_path_config)
else:
    SENTENCE_TRANSFORMER_MODEL = _model_path_config

print(f"📦 向量模型路径: {SENTENCE_TRANSFORMER_MODEL}")


# ==================== 性能监控装饰器 ====================
def monitor_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        print(f"\n⏱️  【{func_name}】开始执行...")
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            print(f"✅ 【{func_name}】完成，耗时: {duration:.2f} ms")
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            print(f"❌ 【{func_name}】失败，耗时: {duration:.2f} ms，错误: {str(e)}")
            raise
    return wrapper


# ==================== 数据库连接池管理 ====================
class DatabasePoolManager:
    _engines = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_engine(cls, db_name: str):
        if db_name not in cls._engines:
            with cls._lock:
                if db_name not in cls._engines:
                    db_config = get_database_config(db_name)
                    if not db_config:
                        raise ValueError(f"数据库配置不存在: {db_name}")
                    cls._engines[db_name] = cls._create_engine(db_config)
                    print(f"🔌 为数据库 {db_name} 创建连接池")
        return cls._engines[db_name]
    
    @staticmethod
    def _create_engine(db_config: Dict):
        db_type = db_config['type']
        
        if db_type == 'postgresql':
            user = quote_plus(str(db_config['user']))
            password = quote_plus(str(db_config['password']))
            host = db_config['host']
            port = db_config.get('port', '5432')
            name = quote_plus(str(db_config['name']))
            sslmode = quote_plus(str(db_config.get('sslmode', 'prefer')))
            url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}?sslmode={sslmode}"
            
            engine = create_engine(
                url,
                pool_size=POOL_SIZE,
                max_overflow=MAX_OVERFLOW,
                pool_pre_ping=POOL_PRE_PING,
                pool_recycle=3600,
                echo=False
            )
            return engine
        
        elif db_type == 'sqlite':
            url = f"sqlite:///{db_config['file_path']}"
            return create_engine(url, pool_size=1, pool_recycle=3600)
        
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")


# ==================== 豆包API客户端 ====================
class DouBaoClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = os.getenv('ARK_MODEL')
        if not self.model:
            raise ValueError('ARK_MODEL environment variable is required')
        self.request_id = 0
    
    @monitor_function
    def generate_text(self, nl_query: str, formatted_tables: str, knowledge_json: Any) -> str:
        formatted_knowledge = self._format_knowledge(knowledge_json)

        print(f"    ├─ 表结构信息长度: {len(formatted_tables)} 字符")
        print(f"    ├─ 知识库示例长度: {len(formatted_knowledge)} 字符")

        self.request_id += 1
        cache_buster = f"\n\n<!-- 请求ID: {self.request_id}, 时间戳: {int(time.time())} -->"

        client = Ark(api_key=self.api_key)

        max_table_length = 8000
        if len(formatted_tables) > max_table_length:
            print(f"    ⚠️ 表结构过长({len(formatted_tables)}字符)，截断到{max_table_length}字符")
            formatted_tables = formatted_tables[:max_table_length] + "\n...(表结构已截断)"

        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"""你是一个专业的SQL查询生成器。你的任务是根据用户的问题生成正确的SQL语句。

## 重要规则：
1. 只输出SQL语句，不要输出任何其他内容
2. 不要输出"请"、"好的"、"以下是"等任何解释性文字
3. 不要使用```sql或```标记
4. SQL语句必须以SELECT或WITH开头
5. 仅使用提供的表结构和知识库进行查询,禁止胡编乱造
6. 如果找不到相关的表或字段，返回: -- 无相关表

## 表结构：
{formatted_tables}

## 知识库：
{formatted_knowledge}
{cache_buster}"""},
                    {"role": "user", "content": f"问题：{nl_query}\n\nSQL："}
                ],
                temperature=0.2,
                top_p=0.95,
                max_tokens=2000
            )
        except Exception as e:
            print(f"    ❌ API调用失败: {str(e)}")
            raise
        
        if hasattr(completion, 'choices'):
            content = completion.choices[0].message.content
            print(f"    📝 AI原始返回: '{content[:200]}...'")

            if content and len(content) > 3:
                cleaned_sql = self._clean_sql(str(content))
                if cleaned_sql:
                    return cleaned_sql

            if content and content.strip().upper().startswith(('SELECT', 'WITH')):
                return content.strip()

            print(f"    ⚠️ AI返回无效内容: {repr(content[:100])}")
            return ""

        raise ValueError('无法解析 AI 返回结果')

    def _clean_sql(self, raw_content: str) -> str:
        if not raw_content:
            return ""
        
        sql = raw_content.strip()
        
        sql = re.sub(r'^```sql\s*\n?', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'^```\s*\n?', '', sql)
        sql = re.sub(r'\n?```$', '', sql)
        
        lines = []
        for line in sql.split('\n'):
            if '--' in line:
                line = line[:line.index('--')]
            line = line.strip()
            if line:
                lines.append(line)
        
        sql = ' '.join(lines) if lines else ""
        
        sql_lower = sql.lower()
        if not (sql_lower.startswith('select') or sql_lower.startswith('with')):
            for line in sql.split('\n'):
                line_clean = line.strip()
                line_lower = line_clean.lower()
                if line_lower.startswith('select') or line_lower.startswith('with'):
                    sql = line_clean
                    break
            else:
                print(f"    ⚠️ 无法提取有效的SQL语句")
                return ""
        
        return sql
    
    def _format_knowledge(self, knowledge_json: List[Dict]) -> str:
        if not knowledge_json:
            return "无可用知识库示例"
        examples = []
        for item in knowledge_json[:5]:
            question = item.get('question', '')
            sql = item.get('sql', '')
            if question and sql:
                examples.append(f"问题: {question}\nSQL: {sql}")
        return "\n\n".join(examples) if examples else "无可用示例"


# ==================== 数据库管理器 ====================
class DatabaseManager:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.engine = DatabasePoolManager.get_engine(db_name)
    
    @monitor_function
    def execute_sql(self, sql: str) -> pd.DataFrame:
        if not sql:
            raise ValueError("SQL语句为空")
        
        clean_sql = self._clean_sql(sql)
        if not clean_sql:
            raise ValueError("SQL语句为空")
        
        if not clean_sql.lower().startswith(('select', 'with')):
            raise ValueError("只允许SELECT查询")
        
        with self.engine.connect() as conn:
            if self.db_name == 'hologres':
                try:
                    conn.execute(text("SET hg_computing_resource = 'serverless'"))
                except:
                    pass
            
            result = conn.execute(text(clean_sql))
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
            
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
        return df
    
    def _clean_sql(self, sql: str) -> str:
        clean_sql = sql.strip()
        clean_sql = re.sub(r'^```sql\s*\n?', '', clean_sql, flags=re.IGNORECASE)
        clean_sql = re.sub(r'^```\s*\n?', '', clean_sql)
        clean_sql = re.sub(r'\n?```$', '', clean_sql)
        
        lines = []
        for line in clean_sql.split('\n'):
            if '--' in line:
                line = line[:line.index('--')]
            line = line.strip()
            if line:
                lines.append(line)
        
        return ' '.join(lines) if lines else ""


# ==================== 知识库管理 ====================
class KnowledgeBase:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.engine = DatabasePoolManager.get_engine(db_name)
        self._ensure_knowledge_table()
        self._ensure_vector_table()
        
        # 缓存目录
        self.cache_dir = Path("./cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 向量模型维度
        self.vector_dim = 384
    
    def _ensure_knowledge_table(self):
        """确保知识库表存在"""
        create_table_sql = """
        CREATE SCHEMA IF NOT EXISTS knowledge;
        CREATE TABLE IF NOT EXISTS knowledge.db_knowledge (
            id SERIAL PRIMARY KEY,
            db_name VARCHAR(50) NOT NULL,
            question TEXT NOT NULL,
            sql TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
                print(f"   ✅ 知识库表已就绪")
        except Exception as e:
            print(f"   ⚠️ 知识库表检查失败: {e}")
    
    def _ensure_vector_table(self):
        """确保向量表存在（使用新表名）"""
        create_table_sql = """
        CREATE SCHEMA IF NOT EXISTS knowledge;
        CREATE TABLE IF NOT EXISTS knowledge.table_embeddings_v2 (
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
                print(f"   ✅ 向量表已就绪: knowledge.table_embeddings_v2")
        except Exception as e:
            print(f"   ⚠️ 向量表创建失败: {e}")
    
    # ========== 向量操作方法 ==========
    
    def save_embeddings_to_holo(self, table_records: List[Dict], embeddings: np.ndarray):
        """保存向量到Hologres（使用新表）"""
        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                for record, embedding in zip(table_records, embeddings):
                    vector_text = record.get('_vector_text', '')
                    text_hash = hashlib.md5(vector_text.encode()).hexdigest()
                    schema_hash = text_hash
                    
                    embedding_list = embedding.tolist()
                    column_info_json = json.dumps(record.get('_columns_json', []))
                    
                    # 删除旧记录
                    delete_sql = """
                    DELETE FROM knowledge.table_embeddings_v2 
                    WHERE db_name = :db_name 
                      AND (schema_name = :schema_name OR (schema_name IS NULL AND :schema_name IS NULL))
                      AND table_name = :table_name
                    """
                    
                    conn.execute(
                        text(delete_sql),
                        {
                            "db_name": self.db_name,
                            "schema_name": record.get('schema'),
                            "table_name": record['table_name']
                        }
                    )
                    
                    # 插入新记录
                    insert_sql = """
                    INSERT INTO knowledge.table_embeddings_v2 
                        (db_name, schema_name, table_name, table_comment, column_info, 
                         vector_text, embedding, text_hash, schema_hash, updated_at)
                    VALUES 
                        (:db_name, :schema_name, :table_name, :table_comment, :column_info, 
                         :vector_text, :embedding, :text_hash, :schema_hash, NOW())
                    """
                    
                    conn.execute(
                        text(insert_sql),
                        {
                            "db_name": self.db_name,
                            "schema_name": record.get('schema'),
                            "table_name": record['table_name'],
                            "table_comment": record.get('table_comment', ''),
                            "column_info": column_info_json,
                            "vector_text": vector_text,
                            "embedding": embedding_list,
                            "text_hash": text_hash,
                            "schema_hash": schema_hash
                        }
                    )
                
                trans.commit()
                print(f"   ✅ 保存 {len(table_records)} 个向量到Hologres")
                return True
                
            except Exception as e:
                trans.rollback()
                print(f"   ❌ 保存向量失败: {e}")
                return False
    
    def vector_search_in_holo(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """在Hologres中使用Proxima索引进行向量检索"""
        
        # 将查询向量转换为 PostgreSQL 数组字符串格式
        embedding_str = '{' + ','.join(str(x) for x in query_embedding) + '}'
        
        # 使用 Hologres Proxima 专有函数进行近似查询
        # 注意：ORDER BY 必须使用 ASC，与 SquaredEuclidean 距离函数配合
        sql = """
        SELECT 
            schema_name,
            table_name,
            table_comment,
            column_info,
            vector_text,
            pm_approx_squared_euclidean_distance(embedding, CAST(:query_embedding AS float4[])) AS similarity_score
        FROM knowledge.table_embeddings_v2
        WHERE db_name = :db_name
        ORDER BY similarity_score ASC
        LIMIT :top_k
        """
        
        try:
            with self.engine.connect() as conn:
                # 对于大数据量查询，可以使用 Serverless 资源
                try:
                    conn.execute(text("SET hg_computing_resource = 'serverless'"))
                except:
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
        """检查Hologres中是否已有向量数据"""
        sql = """
        SELECT COUNT(*) 
        FROM knowledge.table_embeddings_v2 
        WHERE db_name = :db_name
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {"db_name": self.db_name})
            count = result.fetchone()[0]
            return count > 0
    
    def get_holo_vectors_count(self) -> int:
        """获取Hologres中向量数量"""
        sql = """
        SELECT COUNT(*) 
        FROM knowledge.table_embeddings_v2 
        WHERE db_name = :db_name
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {"db_name": self.db_name})
            return result.fetchone()[0]
    
    # ========== 表结构获取方法 ==========
    
    @monitor_function
    def get_table_schema(self) -> Dict:
        print(f"    ├─ 加载表结构: {self.db_name}")
        
        db_config = get_database_config(self.db_name)
        if not db_config:
            return {'tables': {}}
        
        if db_config['type'] == 'postgresql':
            return self._get_postgresql_schema()
        elif db_config['type'] == 'sqlite':
            return self._get_sqlite_schema()
        else:
            return {'tables': {}}
    
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
                        WHEN column_comment != '' THEN column_name || '(' || column_comment || ')'
                        ELSE column_name
                    END,
                    ', '
                    ORDER BY column_order
                ) as columns_text,
                string_agg(
                    CASE 
                        WHEN column_comment != '' THEN '    ' || column_name || ' (' || column_comment || ')'
                        ELSE '    ' || column_name
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
    
    def get_formatted_schema(self, selected_tables: Optional[List[str]] = None) -> str:
        schema_data = self.get_table_schema()
        
        if selected_tables:
            return self._format_selected_tables(schema_data, selected_tables)
        else:
            return self._format_all_tables(schema_data)
    
    def _format_all_tables(self, schema_data: Dict) -> str:
        formatted = []
        
        if 'schemas' in schema_data:
            for schema_name, schema_info in schema_data['schemas'].items():
                for table_name, table_info in schema_info.get('tables', {}).items():
                    columns_formatted = table_info.get('_columns_formatted')
                    table_comment = table_info.get('_table_comment', '')
                    
                    header = f"表: {schema_name}.{table_name}"
                    if table_comment:
                        header += f" (注释: {table_comment})"
                    
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
                header = f"表: {schema_name}.{table_name}"
                if table_comment:
                    header += f" (注释: {table_comment})"
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
    
    # ========== 知识库 CRUD 操作 ==========
    def get_sql_knowledge(self) -> List[Dict]:
        query = """
        SELECT id, question, sql, created_at, updated_at
        FROM knowledge.db_knowledge
        WHERE db_name = :db_name
        ORDER BY id
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"db_name": self.db_name})
                rows = result.fetchall()
                return [
                    {
                        'id': row[0],
                        'question': row[1],
                        'sql': row[2],
                        'created_at': str(row[3]) if row[3] else None,
                        'updated_at': str(row[4]) if row[4] else None
                    }
                    for row in rows
                ]
        except Exception as e:
            print(f"获取知识库失败: {e}")
            return []
    
    def add_knowledge(self, question: str, sql: str) -> Dict:
        if not question or not sql:
            raise ValueError("问题和SQL不能为空")
        
        insert_query = """
        INSERT INTO knowledge.db_knowledge (db_name, question, sql, created_at, updated_at)
        VALUES (:db_name, :question, :sql, NOW(), NOW())
        RETURNING id
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(insert_query),
                    {"db_name": self.db_name, "question": question, "sql": sql}
                )
                new_id = result.fetchone()[0]
                conn.commit()
                return {'id': new_id, 'question': question, 'sql': sql}
        except Exception as e:
            raise ValueError(f"添加知识条目失败: {e}")
    
    def update_knowledge(self, knowledge_id: int, question: str, sql: str) -> bool:
        update_query = """
        UPDATE knowledge.db_knowledge
        SET question = :question, sql = :sql, updated_at = NOW()
        WHERE id = :id AND db_name = :db_name
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(update_query),
                    {"id": knowledge_id, "db_name": self.db_name, "question": question, "sql": sql}
                )
                conn.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"更新知识条目失败: {e}")
            return False
    
    def delete_knowledge(self, knowledge_id: int) -> bool:
        delete_query = """
        DELETE FROM knowledge.db_knowledge
        WHERE id = :id AND db_name = :db_name
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(delete_query),
                    {"id": knowledge_id, "db_name": self.db_name}
                )
                conn.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"删除知识条目失败: {e}")
            return False


# ==================== 向量检索类 ====================
class TableSchemaSearcher:
    _model = None
    _model_lock = threading.Lock()
    
    @classmethod
    def _get_model(cls):
        if cls._model is None:
            with cls._model_lock:
                if cls._model is None:
                    load_start = time.time()
                    cls._model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
                    load_duration = (time.time() - load_start) * 1000
                    print(f"    ├─ 加载向量模型: {load_duration:.2f} ms")
        return cls._model
    
    @classmethod
    @monitor_function
    def search(cls, db_name: str, query: str, top_k: int = 10, 
               kb: KnowledgeBase = None, use_holo_index: bool = True,
               force_rebuild_vectors: bool = False) -> List[Dict]:
        
        if not kb:
            kb = KnowledgeBase(db_name)
        
        print(f"\n    📊 向量检索开始...")
        print(f"    ├─ 查询: {query[:80]}...")
        
        if use_holo_index:
            vectors_count = kb.get_holo_vectors_count()
            
            if vectors_count == 0 or force_rebuild_vectors:
                print(f"    ├─ Hologres中无向量数据或强制重建，开始生成并保存...")
                
                table_records, vector_texts = kb.get_vector_texts()
                
                if not table_records:
                    print(f"    ⚠️ 没有找到任何表")
                    return []
                
                print(f"    ├─ 找到 {len(table_records)} 个表，生成向量中...")
                
                model = cls._get_model()
                
                batch_size = 50
                all_embeddings = []
                for i in range(0, len(vector_texts), batch_size):
                    batch = vector_texts[i:i+batch_size]
                    batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                    all_embeddings.extend(batch_embeddings)
                    print(f"    ├─ 向量生成: {min(i+batch_size, len(vector_texts))}/{len(vector_texts)}")
                
                embeddings = np.array(all_embeddings)
                kb.save_embeddings_to_holo(table_records, embeddings)
                
                print(f"    ├─ 向量已保存到Hologres")
            else:
                print(f"    ├─ 使用Hologres已有向量（{vectors_count}个表）")
        
        model = cls._get_model()
        query_start = time.time()
        query_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        query_duration = (time.time() - query_start) * 1000
        print(f"    ├─ 查询向量生成: {query_duration:.2f} ms")
        
        search_start = time.time()
        results = kb.vector_search_in_holo(query_emb.tolist(), top_k)
        search_duration = (time.time() - search_start) * 1000
        print(f"    ├─ Hologres向量检索: {search_duration:.2f} ms")
        
        if not results:
            print(f"    ⚠️ 向量检索无结果")
            return []
        
        print(f"    📈 检索结果 Top {len(results)}:")
        for r in results:
            table_display = f"{r['schema']}.{r['table_name']}" if r['schema'] else r['table_name']
            print(f"        {r['_rank']}. {r['_similarity_score']:.4f} - {table_display}")
        
        total_duration = (time.time() - query_start) * 1000
        print(f"    ✅ 检索完成，总耗时: {total_duration:.2f} ms")
        
        return results


# ==================== Text2SQL转换器 ====================
class TextToSQLConverter:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.db = DatabaseManager(db_name)
        self.kb = KnowledgeBase(db_name)
        api_key = os.getenv("ARK_API_KEY")
        if not api_key:
            raise ValueError("ARK_API_KEY environment variable is required")
        self.llm = DouBaoClient(api_key=api_key)
    
    @monitor_function
    def execute_nl_query(
        self,
        nl_query: str,
        selected_table: Optional[str] = None,
        use_vector_search: bool = True,
        top_k_tables: int = 10,
        force_rebuild_vectors: bool = False,
    ) -> Tuple[str, pd.DataFrame]:
        print(f"\n📝 查询: {nl_query}")
        
        if selected_table:
            print(f"🎯 指定表模式: {selected_table}")
            formatted_tables = self.kb.get_table_schema_by_name(selected_table)
            if not formatted_tables or formatted_tables.startswith("未找到表"):
                print(f"   ⚠️ 未找到指定表 {selected_table}，尝试使用所有表")
                formatted_tables = self.kb.get_formatted_schema()
            else:
                print(f"   ✅ 只传递了表: {selected_table}")
        elif use_vector_search:
            print(f"🔍 向量检索模式: Top {top_k_tables}")
            best_tables = TableSchemaSearcher.search(
                self.db_name, nl_query, top_k_tables, self.kb, 
                use_holo_index=True, force_rebuild_vectors=force_rebuild_vectors
            )
            
            if not best_tables:
                formatted_tables = self.kb.get_formatted_schema()
                print(f"   ⚠️ 向量检索无结果，使用所有表")
            else:
                selected_names = []
                for t in best_tables:
                    if t.get('schema'):
                        selected_names.append(f"{t['schema']}.{t['table_name']}")
                    else:
                        selected_names.append(t['table_name'])
                formatted_tables = self.kb.get_formatted_schema(selected_names)
                print(f"   ✅ 传递了 {len(selected_names)} 个相关表")
        else:
            print(f"📚 全量模式（所有表结构）")
            formatted_tables = self.kb.get_formatted_schema()
        
        print(f"    ├─ 传递给AI的表结构长度: {len(formatted_tables)} 字符")
        
        knowledge_json = self.kb.get_sql_knowledge()
        sql = self.llm.generate_text(nl_query, formatted_tables, knowledge_json)
        
        if not sql:
            raise ValueError("AI未能生成有效的SQL语句")
        
        result = self.db.execute_sql(sql)
        
        return sql, result


# ==================== 辅助函数 ====================
def precompute_all_embeddings(db_name: str = None, force_rebuild: bool = False):
    """预计算所有数据库的表结构向量并保存到Hologres"""
    
    if db_name:
        db_names = [db_name]
    else:
        from config import get_available_databases
        db_configs = get_available_databases()
        db_names = [db_config['id'] for db_config in db_configs]
    
    for name in db_names:
        print(f"\n{'='*60}")
        print(f"处理数据库: {name}")
        print(f"{'='*60}")
        
        try:
            kb = KnowledgeBase(name)
            
            table_records, vector_texts = kb.get_vector_texts()
            
            if not table_records:
                print(f"   ⚠️ 没有找到任何表")
                continue
            
            print(f"   ├─ 找到 {len(table_records)} 个表")
            
            vectors_count = kb.get_holo_vectors_count()
            if vectors_count > 0 and not force_rebuild:
                print(f"   ├─ Hologres中已有 {vectors_count} 个向量，跳过（使用 --force 强制重建）")
                continue
            
            print(f"   ├─ 加载向量模型...")
            model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
            
            print(f"   ├─ 生成向量中...")
            batch_size = 50
            all_embeddings = []
            for i in range(0, len(vector_texts), batch_size):
                batch = vector_texts[i:i+batch_size]
                batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                print(f"   ├─ 已处理 {min(i+batch_size, len(vector_texts))}/{len(vector_texts)}")
            
            embeddings = np.array(all_embeddings)
            
            kb.save_embeddings_to_holo(table_records, embeddings)
            
            print(f"   ✅ 完成！已保存 {len(table_records)} 个向量到Hologres")
            
        except Exception as e:
            print(f"   ❌ 处理数据库 {name} 时出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "precompute":
        if len(sys.argv) > 2:
            db_name = sys.argv[2]
        else:
            db_name = None
        
        force = len(sys.argv) > 3 and sys.argv[3] == "--force"
        
        precompute_all_embeddings(db_name, force)
    else:
        print("用法:")
        print("  python app_core.py precompute              # 预计算所有数据库向量")
        print("  python app_core.py precompute your_db      # 预计算指定数据库向量")
        print("  python app_core.py precompute your_db --force  # 强制重建")