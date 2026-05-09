# app.py - 优化版（支持中文注释的向量检索）
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import time
import os
import json
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
from functools import wraps
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, inspect, text, Column, Integer, String, Text, TIMESTAMP, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from volcenginesdkarkruntime import Ark
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

from config import (
    KNOWLEDGE_BASE_CONFIGS,
    clear_knowledge_file_override,
    get_knowledge_base_config,
    get_knowledge_override,
    get_available_databases,
    get_database_config,
    set_knowledge_file_override,
)

# ==================== 性能监控装饰器 ====================
def monitor_function(name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__
            start_time = time.time()
            print(f"\n⏱️  【{func_name}】开始执行...")
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            print(f"✅ 【{func_name}】完成，耗时: {duration:.2f} ms")
            return result
        return wrapper
    return decorator

# ==================== 初始化 ====================
app = Flask(__name__)
KB_UPLOAD_DIR = Path(__file__).resolve().parent / "uploads" / "knowledge"
KB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

Base = declarative_base()

# ==================== 数据库模型 ====================
class KnowledgeEntry(Base):
    __tablename__ = 'db_knowledge'
    __table_args__ = {'schema': 'knowledge'}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    db_name = Column(String(50), nullable=False)
    question = Column(Text, nullable=False)
    sql = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

# ==================== 工具函数 ====================
@monitor_function("验证数据库配置")
def validate_db_config(db_name: str) -> Dict:
    db_config = get_database_config(db_name)
    if not db_config:
        raise ValueError(f"数据库配置不存在: {db_name}")
    
    if db_config['type'] == 'mysql':
        missing = [k for k in ('host', 'user', 'password', 'name') if not db_config.get(k)]
        if missing:
            raise ValueError(f"MySQL 数据库配置不完整 ({db_name})，缺少: {', '.join(missing)}")
    elif db_config['type'] == 'sqlite':
        if not db_config.get('file_path'):
            raise ValueError(f"SQLite 数据库文件路径未配置: {db_name}")
    elif db_config['type'] == 'postgresql':
        missing = [k for k in ('host', 'user', 'password', 'name') if not db_config.get(k)]
        if missing:
            raise ValueError(f"PostgreSQL/Hologres 数据库配置不完整 ({db_name})，缺少: {', '.join(missing)}")
    else:
        raise ValueError(f"Unsupported database type: {db_config['type']}")
    
    return db_config

# ==================== 豆包API客户端 ====================
class DouBaoClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = os.getenv('ARK_MODEL') or os.getenv('Model')
        if not self.model:
            raise ValueError('ARK_MODEL or Model environment variable is required')
        
    @monitor_function("豆包API调用")
    def generate_text(self, nl_query: str, tables_json: Any, knowledge_json: Any) -> str:
        format_start = time.time()
        formatted_tables = self._format_table_schema(tables_json)
        formatted_knowledge = self._format_knowledge(knowledge_json)
        format_duration = (time.time() - format_start) * 1000
        print(f"    ├─ 格式化表结构: {format_duration:.2f} ms")
        
        client = Ark(api_key=self.api_key)
        
        api_start = time.time()
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"""你是一个SQL查询生成器。请将自然语言问题转换为精确的SQL查询。

## 可用的表结构信息：
{formatted_tables}

## 参考的知识库示例：
{formatted_knowledge}

## 要求：
1. 只返回SQL语句，不要包含任何解释或markdown格式
2. 确保SQL语法正确
3. 使用标准SQL语法
4. 字段名必须使用上面提供的准确字段名
5. 如果字段不存在，不要使用该字段
6. 返回空字符串如果无法生成有效SQL
                """},
                {"role": "user", "content": nl_query}
            ]
        )
        api_duration = (time.time() - api_start) * 1000
        print(f"    ├─ API请求: {api_duration:.2f} ms")
        
        if hasattr(completion, 'choices'):
            choice = completion.choices[0]
            if hasattr(choice, 'message') and getattr(choice.message, 'content', None) is not None:
                return str(choice.message.content)
        raise ValueError('无法解析 AI 返回结果')
    
    def _format_table_schema(self, tables_json: Dict) -> str:
        if isinstance(tables_json, str):
            tables_json = json.loads(tables_json)
        
        formatted = []
        if 'schemas' in tables_json:
            for schema_name, schema_info in tables_json['schemas'].items():
                for table_name, columns in schema_info.get('tables', {}).items():
                    col_list = []
                    for col_name, col_comment in columns.items():
                        if col_comment and col_comment != '_table_comment':
                            col_list.append(f"    {col_name}({col_comment})")
                        else:
                            col_list.append(f"    {col_name}")
                    formatted.append(f"表: {schema_name}.{table_name}\n列:\n" + "\n".join(col_list))
        elif 'tables' in tables_json:
            for table_name, columns in tables_json['tables'].items():
                col_list = []
                for col_name, col_comment in columns.items():
                    if col_comment and col_comment != '_table_comment':
                        col_list.append(f"    {col_name}({col_comment})")
                    else:
                        col_list.append(f"    {col_name}")
                formatted.append(f"表: {table_name}\n列:\n" + "\n".join(col_list))
        return "\n\n".join(formatted) if formatted else "无可用表结构"
    
    def _format_knowledge(self, knowledge_json: List[Dict]) -> str:
        if not knowledge_json:
            return "无可用知识库示例"
        examples = []
        for item in knowledge_json[:5]:
            question = item.get('问题', '')
            sql = item.get('sql', '')
            if question and sql:
                examples.append(f"问题: {question}\nSQL: {sql}")
        return "\n\n".join(examples) if examples else "无可用示例"

# ==================== 数据库管理器 ====================
class DatabaseManager:
    def __init__(self, db_name: str = 'book'):
        self.db_name = db_name
        self.engine = self._create_engine()
        
    @monitor_function("创建数据库引擎")
    def _create_engine(self):
        if not self.db_name:
            self.db_name = 'db'
        db_config = get_database_config(self.db_name)
        if not db_config:
            raise ValueError(f"数据库 '{self.db_name}' 不存在")
        return self._create_engine_static(db_config)
    
    @staticmethod
    def _create_engine_static(db_config):
        if db_config['type'] == 'mysql':
            url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}"
        elif db_config['type'] == 'sqlite':
            url = f"sqlite:///{db_config['file_path']}"
        elif db_config['type'] == 'postgresql':
            user = quote_plus(str(db_config['user']))
            password = quote_plus(str(db_config['password']))
            host = db_config['host']
            port = db_config.get('port', '80')
            name = quote_plus(str(db_config['name']))
            sslmode = quote_plus(str(db_config.get('sslmode', 'prefer')))
            url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}?sslmode={sslmode}"
        else:
            raise ValueError(f"Unsupported database type: {db_config['type']}")
        return create_engine(url)
    
    @monitor_function("执行SQL查询")
    def execute_sql(self, sql: str) -> pd.DataFrame:
        clean_sql = sql.replace('`', '').replace('sql\n', '')
        if not clean_sql.lower().startswith(('select', 'with')):
            raise ValueError(f"Only SELECT queries are allowed :{clean_sql}")
        
        with self.engine.connect() as conn:
            result = conn.execute(text(clean_sql))
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or 'datetime' in str(df[col].dtype):
                    df[col] = df[col].astype(str)
        return df

# ==================== 知识库管理（含中文注释） ====================
class KnowledgeBase:
    def __init__(self, db_name: str = 'box'):
        self.db_name = db_name
        self.use_db = db_name == 'hologres'
        kb_config = get_knowledge_base_config(db_name)
        self.file_path = kb_config.get('file_path') if kb_config else None
        self.sheet_name = kb_config.get('sheet_name') if kb_config else '知识库'
        
        self._schema_cache = None
        self._cache_timestamp = None
        self._cache_ttl = 300
    
    @monitor_function("获取SQL知识库")
    def get_sql_knowledge(self) -> List[Dict]:
        if self.use_db:
            return self._get_from_db()
        return self._get_from_excel()
    
    def _get_from_excel(self) -> List[Dict]:
        if not self.file_path:
            return []
        try:
            knowledge_table = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            return knowledge_table.to_dict("records")
        except Exception as e:
            print(f"加载知识库失败: {str(e)}")
            return []
    
    def _get_from_db(self) -> List[Dict]:
        try:
            db_config = get_database_config(self.db_name)
            engine = DatabaseManager._create_engine_static(db_config)
            Session = sessionmaker(bind=engine)
            session = Session()
            entries = session.query(KnowledgeEntry).filter_by(db_name=self.db_name).all()
            result = [{'问题': e.question, 'sql': e.sql, 'id': e.id} for e in entries]
            session.close()
            return result
        except Exception as e:
            print(f"从数据库加载知识库失败: {str(e)}")
            return []
    
    @monitor_function("获取数据库表结构")
    def get_table_schema(self) -> Dict:
        if self._schema_cache and self._cache_timestamp:
            if time.time() - self._cache_timestamp < self._cache_ttl:
                print(f"    ├─ 使用缓存的表结构")
                return self._schema_cache
        
        result = self._get_table_schema_from_db()
        self._schema_cache = result
        self._cache_timestamp = time.time()
        return result

    @monitor_function("从数据库获取表结构详情")
    def _get_table_schema_from_db(self) -> Dict:
        db_config = get_database_config(self.db_name)
        if not db_config:
            return {'tables': {}}
        
        try:
            engine = DatabaseManager._create_engine_static(db_config)
            db_type = db_config['type']
            
            if db_type == 'postgresql':
                return self._get_postgresql_schema_with_comments(engine)
            elif db_type == 'mysql':
                return self._get_mysql_schema_with_comments(engine)
            else:
                return self._get_sqlite_schema(engine)
        except Exception as e:
            print(f"从数据库获取表结构失败: {str(e)}")
            return {'tables': {}}
    
    def _get_postgresql_schema_with_comments(self, engine) -> Dict:
        """获取PostgreSQL表结构，包含中文注释"""
        print(f"    ├─ 获取PostgreSQL表结构（含注释）...")
        
        # 查询表注释和列注释
        query = """
        SELECT 
            t.table_schema,
            t.table_name,
            c.column_name,
            COALESCE(pg_catalog.col_description(
                (t.table_schema||'.'||t.table_name)::regclass::oid,
                c.ordinal_position
            ), '') as column_comment,
            COALESCE(pg_catalog.obj_description(
                (t.table_schema||'.'||t.table_name)::regclass::oid,
                'pg_class'
            ), '') as table_comment
        FROM information_schema.tables t
        JOIN information_schema.columns c 
            ON t.table_schema = c.table_schema 
            AND t.table_name = c.table_name
        WHERE t.table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_schema, t.table_name, c.ordinal_position;
        """
        
        query_start = time.time()
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
        query_duration = (time.time() - query_start) * 1000
        print(f"    ├─ 查询所有表和列（含注释）: {query_duration:.2f} ms")
        
        build_start = time.time()
        schemas = {}
        for row in rows:
            schema_name = row[0]
            table_name = row[1]
            column_name = row[2]
            column_comment = row[3]
            table_comment = row[4]
            
            if schema_name not in schemas:
                schemas[schema_name] = {'tables': {}}
            
            if table_name not in schemas[schema_name]['tables']:
                schemas[schema_name]['tables'][table_name] = {}
                # 存储表注释
                if table_comment:
                    schemas[schema_name]['tables'][table_name]['_table_comment'] = table_comment
            
            schemas[schema_name]['tables'][table_name][column_name] = column_comment
        
        build_duration = (time.time() - build_start) * 1000
        print(f"    ├─ 构建JSON结构: {build_duration:.2f} ms")
        return {'schemas': schemas}
    
    def _get_mysql_schema_with_comments(self, engine) -> Dict:
        """获取MySQL表结构，包含中文注释"""
        print(f"    ├─ 获取MySQL表结构（含注释）...")
        
        with engine.connect() as conn:
            db_name_result = conn.execute(text("SELECT DATABASE()")).fetchone()
            current_db = db_name_result[0] if db_name_result else None
        
        query = """
        SELECT 
            t.table_name,
            c.column_name,
            COALESCE(c.column_comment, '') as column_comment,
            COALESCE(t.table_comment, '') as table_comment
        FROM information_schema.tables t
        JOIN information_schema.columns c 
            ON t.table_schema = c.table_schema 
            AND t.table_name = c.table_name
        WHERE t.table_schema = :db_name
            AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_name, c.ordinal_position;
        """
        
        query_start = time.time()
        with engine.connect() as conn:
            result = conn.execute(text(query), {'db_name': current_db})
            rows = result.fetchall()
        query_duration = (time.time() - query_start) * 1000
        print(f"    ├─ 查询所有表和列（含注释）: {query_duration:.2f} ms")
        
        build_start = time.time()
        tables = {}
        for row in rows:
            table_name = row[0]
            column_name = row[1]
            column_comment = row[2]
            table_comment = row[3]
            
            if table_name not in tables:
                tables[table_name] = {}
                if table_comment:
                    tables[table_name]['_table_comment'] = table_comment
            
            tables[table_name][column_name] = column_comment
        
        build_duration = (time.time() - build_start) * 1000
        print(f"    ├─ 构建JSON结构: {build_duration:.2f} ms")
        return {'tables': tables}
    
    def _get_sqlite_schema(self, engine) -> Dict:
        print(f"    ├─ 获取SQLite表结构...")
        
        query_start = time.time()
        with engine.connect() as conn:
            tables_result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
            table_names = [row[0] for row in tables_result.fetchall()]
            
            tables = {}
            for table_name in table_names:
                pragma_result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = {row[1]: '' for row in pragma_result.fetchall()}
                tables[table_name] = columns
        
        query_duration = (time.time() - query_start) * 1000
        print(f"    ├─ 查询所有表结构: {query_duration:.2f} ms")
        return {'tables': tables}
    
    def get_table_schema_for_table(self, table_name: str) -> Dict:
        schema_data = self.get_table_schema()
        schema_name = None
        simple_table_name = table_name
        if '.' in table_name:
            schema_name, simple_table_name = table_name.split('.', 1)

        if 'schemas' in schema_data:
            if schema_name and schema_name in schema_data['schemas']:
                schema_info = schema_data['schemas'][schema_name]
                if simple_table_name in schema_info.get('tables', {}):
                    return {'schemas': {schema_name: {'tables': {simple_table_name: schema_info['tables'][simple_table_name]}}}}
            else:
                for schema, schema_info in schema_data['schemas'].items():
                    if simple_table_name in schema_info.get('tables', {}):
                        return {'schemas': {schema: {'tables': {simple_table_name: schema_info['tables'][simple_table_name]}}}}
        else:
            tables = schema_data.get('tables', {})
            if simple_table_name in tables:
                return {'tables': {simple_table_name: tables[simple_table_name]}}
        return {'tables': {}}
    
    @monitor_function("获取表列表")
    def get_table_list(self) -> List[Dict[str, Any]]:
        schema_data = self.get_table_schema()
        
        table_list = []
        if 'schemas' in schema_data:
            for schema, schema_info in schema_data['schemas'].items():
                for table_name, columns in schema_info.get('tables', {}).items():
                    # 过滤掉内部字段
                    cols = [col for col in columns.keys() if not col.startswith('_')]
                    table_list.append({
                        'schema': schema,
                        'table_name': table_name,
                        'columns': cols,
                        'column_count': len(cols),
                        'table_comment': columns.get('_table_comment', '')
                    })
        else:
            for table_name, columns in schema_data.get('tables', {}).items():
                cols = [col for col in columns.keys() if not col.startswith('_')]
                table_list.append({
                    'schema': None,
                    'table_name': table_name,
                    'columns': cols,
                    'column_count': len(cols),
                    'table_comment': columns.get('_table_comment', '')
                })
        
        print(f"\n    📋 数据库中的所有表 ({len(table_list)} 个):")
        print(f"    {'='*60}")
        for i, table in enumerate(table_list[:20], 1):
            table_display = f"{table['schema']}.{table['table_name']}" if table['schema'] else table['table_name']
            comment_info = f" [注释: {table['table_comment']}]" if table['table_comment'] else ""
            print(f"    {i}. {table_display}{comment_info} (列数: {table['column_count']})")
            cols_preview = ', '.join(table['columns'][:5])
            if table['column_count'] > 5:
                cols_preview += f"... (共{table['column_count']}列)"
            print(f"       列: {cols_preview}")
        
        if len(table_list) > 20:
            print(f"    ... 还有 {len(table_list) - 20} 个表未显示")
        print(f"    {'='*60}\n")
        
        return table_list

# ==================== 向量检索（支持中文注释） ====================
class TableSchemaSearcher:
    _model = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            load_start = time.time()
            # 使用多语言模型，对中文更好
            cls._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            load_duration = (time.time() - load_start) * 1000
            print(f"    ├─ 加载向量模型: {load_duration:.2f} ms")
        return cls._model

    @classmethod
    def _build_text_with_comments(cls, record: Dict[str, Any], schema_data: Dict = None) -> str:
        """构建用于向量检索的文本，包含中文注释"""
        table_name = record.get('table_name')
        schema = record.get('schema')
        columns = record.get('columns') or []
        table_comment = record.get('table_comment', '')
        
        # 构建列信息，包含中文注释
        column_info = []
        if schema_data and 'schemas' in schema_data:
            for schema_name, schema_info in schema_data['schemas'].items():
                if schema_name == schema and table_name in schema_info.get('tables', {}):
                    cols_data = schema_info['tables'][table_name]
                    for col in columns:
                        col_comment = cols_data.get(col, '')
                        if col_comment and not col.startswith('_'):
                            column_info.append(f"{col}({col_comment})")
                        else:
                            column_info.append(col)
                    break
        elif schema_data and 'tables' in schema_data:
            if table_name in schema_data['tables']:
                cols_data = schema_data['tables'][table_name]
                for col in columns:
                    col_comment = cols_data.get(col, '')
                    if col_comment and not col.startswith('_'):
                        column_info.append(f"{col}({col_comment})")
                    else:
                        column_info.append(col)
        
        if not column_info:
            column_info = columns[:30]
        
        # 构建完整的表描述
        table_display = f"{schema}.{table_name}" if schema else table_name
        description_parts = [f"表名: {table_display}"]
        
        if table_comment:
            description_parts.append(f"表注释: {table_comment}")
        
        description_parts.append(f"列: {', '.join(column_info[:30])}")
        
        if len(columns) > 30:
            description_parts.append(f"... 共{len(columns)}列")
        
        return ' '.join(description_parts)

    @classmethod
    def _encode(cls, texts: List[str]):
        encode_start = time.time()
        result = cls._get_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        encode_duration = (time.time() - encode_start) * 1000
        print(f"    ├─ 向量编码: {encode_duration:.2f} ms (文本数: {len(texts)})")
        return result

    @classmethod
    @monitor_function("向量检索表")
    def search_top_tables(cls, db_name: str, table_records: List[Dict[str, Any]], query: str, 
                          schema_data: Dict = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """返回前top_k个相似表，使用中文注释"""
        if not table_records:
            return []
        
        print(f"\n    📊 开始向量检索，共 {len(table_records)} 个表候选")
        print(f"    📝 查询语句: {query}")
        
        # 构建包含中文注释的文本
        texts = [cls._build_text_with_comments(record, schema_data) for record in table_records]
        
        # 打印第一个表的文本样例
        if texts:
            print(f"    📝 文本样例: {texts[0][:200]}...")
        
        # 编码
        embeddings = cls._encode(texts)
        query_emb = cls._encode([query])[0]
        
        # 计算相似度
        table_norms = np.linalg.norm(embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        
        if query_norm < 1e-8 or table_norms.sum() == 0:
            print(f"    ⚠️ 向量计算异常，返回前{top_k}个表")
            return table_records[:top_k]
        
        similarities = (embeddings @ query_emb) / (table_norms * query_norm + 1e-10)
        top_indices = np.argsort(-similarities)[:top_k]
        
        print(f"\n    {'='*60}")
        print(f"    📈 向量相似度排名 (Top {min(top_k, len(top_indices))}):")
        print(f"    {'='*60}")
        
        results = []
        for i, idx in enumerate(top_indices, 1):
            record = table_records[idx]
            similarity = similarities[idx]
            
            table_display = f"{record.get('schema', '')}.{record['table_name']}" if record.get('schema') else record['table_name']
            comment_info = f" [注释: {record.get('table_comment', '')}]" if record.get('table_comment') else ""
            
            # 获取列的中文注释
            col_comments = []
            if schema_data and 'schemas' in schema_data:
                for schema_name, schema_info in schema_data['schemas'].items():
                    if schema_name == record.get('schema') and record['table_name'] in schema_info.get('tables', {}):
                        cols = schema_info['tables'][record['table_name']]
                        for col in record.get('columns', [])[:5]:
                            col_comment = cols.get(col, '')
                            if col_comment and not col.startswith('_'):
                                col_comments.append(f"{col}({col_comment})")
                            else:
                                col_comments.append(col)
                        break
            elif schema_data and 'tables' in schema_data:
                if record['table_name'] in schema_data['tables']:
                    cols = schema_data['tables'][record['table_name']]
                    for col in record.get('columns', [])[:5]:
                        col_comment = cols.get(col, '')
                        if col_comment and not col.startswith('_'):
                            col_comments.append(f"{col}({col_comment})")
                        else:
                            col_comments.append(col)
            
            if not col_comments:
                col_comments = record.get('columns', [])[:5]
            
            columns_preview = ', '.join(col_comments)
            if len(record.get('columns', [])) > 5:
                columns_preview += f"... (共{len(record.get('columns', []))}列)"
            
            print(f"    {i}. 相似度: {similarity:.4f} | 表名: {table_display}{comment_info}")
            print(f"       列: {columns_preview}")
            
            results.append(record)
            record['_similarity_score'] = float(similarity)
            record['_rank'] = i
        
        print(f"    {'='*60}\n")
        
        return results[:top_k]

# ==================== Text2SQL转换器 ====================
class TextToSQLConverter:
    def __init__(self, db_name: str = 'book'):
        self.db_name = db_name
        self.db = DatabaseManager(db_name)
        self.kb = KnowledgeBase(db_name)
        api_key = os.getenv("ARK_API_KEY") or ""
        if not api_key:
            raise ValueError("ARK_API_KEY environment variable is required")
        self.llm = DouBaoClient(api_key=api_key)
        self._schema_data = None  # 缓存完整的schema数据
    
    def _get_cache_key(self, nl_query: str, selected_table: Optional[str] = None, table_schema_hash: Optional[str] = None) -> str:
        content = f"{nl_query}|{selected_table or 'None'}|{table_schema_hash or 'None'}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_sql(self, cache_key: str) -> Optional[str]:
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if time.time() - cached_data['timestamp'] < 3600:
                        print(f"    ├─ 命中缓存")
                        return cached_data['sql']
            except Exception:
                pass
        return None
    
    def _cache_sql(self, cache_key: str, sql: str):
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'sql': sql, 'timestamp': time.time()}, f)
        except Exception:
            pass

    @monitor_function("Text2SQL转换总流程")
    def execute_nl_query(
        self,
        nl_query: str,
        selected_table: Optional[str] = None,
        use_vector_search: bool = True,
        top_k_tables: int = 10,
    ) -> Tuple[str, pd.DataFrame]:
        print(f"\n📝 自然语言查询: {nl_query}")
        
        schema_start = time.time()
        if selected_table:
            print(f"\n📊 使用指定表: {selected_table}")
            tables_json = self.kb.get_table_schema_for_table(selected_table)
        elif use_vector_search:
            print(f"\n🔍 使用向量检索相关表")
            tables_json = self._get_relevant_tables_schema(nl_query, top_k=top_k_tables)
        else:
            print(f"\n📚 使用全部表结构")
            tables_json = self.kb.get_table_schema()
        schema_duration = (time.time() - schema_start) * 1000
        print(f"    ├─ 获取表结构: {schema_duration:.2f} ms")
        
        table_schema_str = json.dumps(tables_json, sort_keys=True)
        table_schema_hash = hashlib.md5(table_schema_str.encode()).hexdigest()
        
        cache_key = self._get_cache_key(nl_query, selected_table, table_schema_hash)
        cached_sql = self._get_cached_sql(cache_key)
        
        if cached_sql:
            return cached_sql, self.db.execute_sql(cached_sql)
        
        knowledge_json = self.kb.get_sql_knowledge()
        sql = self.llm.generate_text(nl_query, tables_json, knowledge_json)
        
        self._cache_sql(cache_key, sql)
        result = self.db.execute_sql(sql)
        
        return sql, result

    @monitor_function("向量检索相关表")
    def _get_relevant_tables_schema(self, nl_query: str, top_k: int = 10) -> Dict:
        print(f"\n    🔍 目标: 找到与查询最相关的 {top_k} 个表")
        
        # 获取完整的schema数据（含注释）
        self._schema_data = self.kb.get_table_schema()
        table_list = self.kb.get_table_list()
        
        best_tables = TableSchemaSearcher.search_top_tables(
            self.db_name, table_list, nl_query, schema_data=self._schema_data, top_k=top_k
        )
        
        if not best_tables:
            print(f"    ⚠️ 未找到相关表，返回全部表结构")
            return self._schema_data

        print(f"\n    ✅ 最终选中 {len(best_tables)} 个相关表用于SQL生成:")
        print(f"    {'='*60}")
        for i, table in enumerate(best_tables, 1):
            table_display = f"{table['schema']}.{table['table_name']}" if table.get('schema') else table['table_name']
            similarity = table.get('_similarity_score', 0)
            comment_info = f" [注释: {table.get('table_comment', '')}]" if table.get('table_comment') else ""
            print(f"    {i}. {table_display}{comment_info} (相似度: {similarity:.4f})")
        print(f"    {'='*60}\n")

        # 从完整schema中提取选中的表
        if any(table.get('schema') for table in best_tables):
            result = {'schemas': {}}
            for table in best_tables:
                schema_name = table.get('schema') or 'public'
                result['schemas'].setdefault(schema_name, {'tables': {}})
                
                # 从完整schema中获取带注释的列信息
                if 'schemas' in self._schema_data and schema_name in self._schema_data['schemas']:
                    if table['table_name'] in self._schema_data['schemas'][schema_name]['tables']:
                        result['schemas'][schema_name]['tables'][table['table_name']] = \
                            self._schema_data['schemas'][schema_name]['tables'][table['table_name']]
                    else:
                        result['schemas'][schema_name]['tables'][table['table_name']] = {col: '' for col in table['columns']}
                else:
                    result['schemas'][schema_name]['tables'][table['table_name']] = {col: '' for col in table['columns']}
            return result

        result = {'tables': {}}
        for table in best_tables:
            if 'tables' in self._schema_data and table['table_name'] in self._schema_data['tables']:
                result['tables'][table['table_name']] = self._schema_data['tables'][table['table_name']]
            else:
                result['tables'][table['table_name']] = {col: '' for col in table['columns']}
        return result

# ==================== 路由（保持不变） ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/knowledge-management')
def knowledge_management():
    return render_template('knowledge_management.html')

@app.route('/get_databases', methods=['GET'])
def get_databases():
    try:
        databases = get_available_databases()
        return jsonify({'databases': databases, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/knowledge/sources', methods=['GET'])
def knowledge_sources():
    return jsonify({'db_ids': list(KNOWLEDGE_BASE_CONFIGS.keys())})

@app.route('/api/db_tables', methods=['GET'])
def get_db_tables():
    db_name = (request.args.get('db_name') or '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    try:
        kb = KnowledgeBase(db_name)
        table_list = kb.get_table_list()
        return jsonify({'status': 'success', 'tables': table_list})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/knowledge/status', methods=['GET'])
def knowledge_status():
    db_name = (request.args.get('db_name') or '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    if db_name not in KNOWLEDGE_BASE_CONFIGS:
        return jsonify({'status': 'success', 'unsupported': True, 'db_name': db_name})
    default_cfg = KNOWLEDGE_BASE_CONFIGS[db_name]
    eff = get_knowledge_base_config(db_name)
    ov = get_knowledge_override(db_name)
    return jsonify({
        'status': 'success',
        'db_name': db_name,
        'default_file_path': default_cfg.get('file_path'),
        'from_upload': bool(ov and ov.get('file_path')),
    })

@app.route('/api/knowledge/upload', methods=['POST'])
def knowledge_upload():
    if 'file' not in request.files:
        return jsonify({'error': '请选择要上传的文件'}), 400
    db_name = (request.form.get('db_name') or '').strip()
    sheet_name = (request.form.get('sheet_name') or '知识库').strip() or '知识库'
    if not db_name or db_name not in KNOWLEDGE_BASE_CONFIGS:
        return jsonify({'error': '无效的数据源'}), 400
    f = request.files['file']
    ext = Path(f.filename).suffix.lower()
    if ext not in ('.xlsx', '.xls'):
        return jsonify({'error': '仅支持 .xlsx / .xls'}), 400
    
    try:
        df = pd.read_excel(f, sheet_name=sheet_name)
    except Exception as e:
        return jsonify({'error': f'无法读取工作表: {e}'}), 400
    
    if db_name == 'hologres':
        try:
            db_config = get_database_config(db_name)
            engine = DatabaseManager._create_engine_static(db_config)
            inserted_count = 0
            with engine.connect() as conn:
                for _, row in df.iterrows():
                    q = str(row.get('问题', '')).strip()
                    s = str(row.get('sql', row.get('SQL', ''))).strip()
                    if q and s and q != 'nan' and s != 'nan':
                        conn.execute(
                            text("INSERT INTO knowledge.db_knowledge (db_name, question, sql) VALUES (:db_name, :question, :sql)"),
                            {'db_name': db_name, 'question': q, 'sql': s}
                        )
                        inserted_count += 1
                conn.commit()
            return jsonify({'status': 'success', 'inserted_count': inserted_count})
        except Exception as e:
            return jsonify({'error': f'上传失败: {str(e)}'}), 500
    else:
        dest = KB_UPLOAD_DIR / f"{db_name}_knowledge{ext}"
        f.save(str(dest))
        set_knowledge_file_override(db_name, str(dest.resolve()), sheet_name)
        return jsonify({'status': 'success', 'row_count': len(df)})

@app.route('/api/knowledge/reset', methods=['POST'])
def knowledge_reset():
    data = request.get_json(silent=True) or {}
    db_name = (data.get('db_name') or '').strip()
    if not db_name or db_name not in KNOWLEDGE_BASE_CONFIGS:
        return jsonify({'error': 'invalid db_name'}), 400
    clear_knowledge_file_override(db_name)
    return jsonify({'status': 'success', 'message': '已恢复默认知识库配置'})

@app.route('/execute_sql', methods=['POST'])
def execute_sql():
    try:
        data = request.get_json()
        sql = data.get('sql')
        db_name = data.get('db_name')
        if not sql or not db_name:
            return jsonify({'error': 'Missing sql or db_name'}), 400
        validate_db_config(db_name)
        db = DatabaseManager(db_name)
        result = db.execute_sql(sql)
        return jsonify({'sql_result': result.to_dict(orient='records'), 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/execute_nl_query', methods=['POST'])
def handle_nl_query():
    try:
        data = request.get_json()
        if not data or 'nl_query' not in data:
            return jsonify({'error': 'Missing nl_query'}), 400
        
        db_name = data.get('db_name')
        if not db_name:
            return jsonify({'error': 'Missing db_name'}), 400

        validate_db_config(db_name)
        selected_table = (data.get('selected_table') or '').strip() or None
        use_vector_search = data.get('use_vector_search', True)
        
        sql, result = TextToSQLConverter(db_name).execute_nl_query(
            data['nl_query'],
            selected_table=selected_table,
            use_vector_search=use_vector_search,
            top_k_tables=10
        )
        
        return jsonify({'generated_sql': sql, 'sql_result': result.to_dict(orient='records'), 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/knowledge', methods=['GET'])
def get_knowledge():
    db_name = request.args.get('db_name', 'hologres')
    if db_name != 'hologres':
        return jsonify({'error': 'Only hologres database supports knowledge management'}), 400
    try:
        kb = KnowledgeBase(db_name)
        knowledge = kb.get_sql_knowledge()
        return jsonify({'knowledge': knowledge, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/knowledge', methods=['POST'])
def add_knowledge():
    data = request.get_json()
    db_name = data.get('db_name', 'hologres')
    question = data.get('question')
    sql = data.get('sql')
    if not question or not sql:
        return jsonify({'error': 'question and sql are required'}), 400
    if db_name != 'hologres':
        return jsonify({'error': 'Only hologres database supports knowledge management'}), 400
    try:
        db_config = get_database_config(db_name)
        engine = DatabaseManager._create_engine_static(db_config)
        with engine.connect() as conn:
            insert_sql = text("INSERT INTO knowledge.db_knowledge (db_name, question, sql) VALUES (:db_name, :question, :sql)")
            result = conn.execute(insert_sql, {'db_name': db_name, 'question': question, 'sql': sql})
            conn.commit()
            entry_id = result.lastrowid if hasattr(result, 'lastrowid') else None
        return jsonify({'id': entry_id, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/knowledge/<int:entry_id>', methods=['PUT'])
def update_knowledge(entry_id):
    data = request.get_json()
    question = data.get('question')
    sql = data.get('sql')
    if not question or not sql:
        return jsonify({'error': 'question and sql are required'}), 400
    try:
        db_config = get_database_config('hologres')
        engine = DatabaseManager._create_engine_static(db_config)
        Session = sessionmaker(bind=engine)
        session = Session()
        entry = session.query(KnowledgeEntry).filter_by(id=entry_id).first()
        if not entry:
            return jsonify({'error': 'Knowledge entry not found'}), 404
        entry.question = question
        entry.sql = sql
        session.commit()
        session.close()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/knowledge/<int:entry_id>', methods=['DELETE'])
def delete_knowledge(entry_id):
    try:
        db_config = get_database_config('hologres')
        engine = DatabaseManager._create_engine_static(db_config)
        Session = sessionmaker(bind=engine)
        session = Session()
        entry = session.query(KnowledgeEntry).filter_by(id=entry_id).first()
        if not entry:
            return jsonify({'error': 'Knowledge entry not found'}), 404
        session.delete(entry)
        session.commit()
        session.close()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/init_knowledge_table', methods=['POST'])
def init_table():
    init_knowledge_table('hologres')
    return jsonify({'status': 'success'})

# ==================== 调试接口 ====================
@app.route('/api/test_vector_search', methods=['POST'])
def test_vector_search():
    """测试向量检索，返回相似度排名"""
    try:
        data = request.get_json()
        db_name = data.get('db_name')
        query = data.get('query')
        top_k = data.get('top_k', 10)
        
        if not db_name or not query:
            return jsonify({'error': 'Missing db_name or query'}), 400
        
        kb = KnowledgeBase(db_name)
        schema_data = kb.get_table_schema()
        table_list = kb.get_table_list()
        
        results = TableSchemaSearcher.search_top_tables(
            db_name, table_list, query, schema_data=schema_data, top_k=top_k
        )
        
        return jsonify({
            'status': 'success',
            'query': query,
            'top_tables': [
                {
                    'rank': t.get('_rank', i+1),
                    'similarity': t.get('_similarity_score', 0),
                    'table_name': f"{t.get('schema', '')}.{t['table_name']}" if t.get('schema') else t['table_name'],
                    'table_comment': t.get('table_comment', ''),
                    'columns': t['columns'][:10],
                    'column_count': len(t['columns'])
                }
                for i, t in enumerate(results)
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/validate_table_fields', methods=['POST'])
def validate_table_fields():
    """验证表是否包含指定字段"""
    try:
        data = request.get_json()
        db_name = data.get('db_name')
        table_name = data.get('table_name')
        fields = data.get('fields', [])
        
        if not db_name or not table_name:
            return jsonify({'error': 'Missing db_name or table_name'}), 400
        
        kb = KnowledgeBase(db_name)
        schema = kb.get_table_schema()
        
        found_table = None
        if 'schemas' in schema:
            for schema_name, schema_info in schema['schemas'].items():
                for t_name, columns in schema_info.get('tables', {}).items():
                    if f"{schema_name}.{t_name}" == table_name or t_name == table_name:
                        found_table = {'schema': schema_name, 'table': t_name, 'columns': columns}
                        break
                if found_table:
                    break
        else:
            for t_name, columns in schema.get('tables', {}).items():
                if t_name == table_name:
                    found_table = {'schema': None, 'table': t_name, 'columns': columns}
                    break
        
        if not found_table:
            return jsonify({'error': f'Table {table_name} not found'}), 404
        
        available_fields = [col for col in found_table['columns'].keys() if not col.startswith('_')]
        missing_fields = [f for f in fields if f not in available_fields]
        existing_fields = [f for f in fields if f in available_fields]
        
        return jsonify({
            'status': 'success',
            'table_name': table_name,
            'available_fields': available_fields,
            'existing_fields': existing_fields,
            'missing_fields': missing_fields,
            'field_count': len(available_fields)
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# ==================== 启动应用 ====================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 启动 Flask 应用 (支持中文注释向量检索)")
    print("="*60)
    
    old_caches = 0
    for cache_file in CACHE_DIR.glob("*.pkl"):
        try:
            if time.time() - cache_file.stat().st_mtime > 86400:
                cache_file.unlink()
                old_caches += 1
        except Exception:
            pass
    
    print(f"📁 缓存目录: {CACHE_DIR}")
    print(f"📁 上传目录: {KB_UPLOAD_DIR}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)