# core/utils.py - 通用工具：缓存、限流、重试、SQL 清洗、配置常量
import os
import re
import time
import threading
from pathlib import Path
from collections import OrderedDict
from functools import wraps
from typing import List

from dotenv import load_dotenv

load_dotenv()

# ==================== 配置 ====================
POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '5'))
MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '10'))
POOL_PRE_PING = os.getenv('DB_POOL_PRE_PING', 'true').lower() == 'true'

# 向量模型路径配置（支持相对路径和绝对路径）
_model_path_config = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')

# 解析路径：相对路径基于当前文件所在目录的父目录（即项目根 text_to_sql/）
if _model_path_config.startswith('../') or _model_path_config.startswith('./'):
    _current_file_dir = Path(__file__).parent.parent.absolute()
    SENTENCE_TRANSFORMER_MODEL = str(_current_file_dir / _model_path_config)
elif not _model_path_config.startswith('/') and not _model_path_config.startswith('http'):
    _current_file_dir = Path(__file__).parent.parent.absolute()
    SENTENCE_TRANSFORMER_MODEL = str(_current_file_dir / _model_path_config)
else:
    SENTENCE_TRANSFORMER_MODEL = _model_path_config

print(f"📦 向量模型路径: {SENTENCE_TRANSFORMER_MODEL}")

# 向量嵌入方式选择：'local'（本地 SentenceTransformer）或 'api'（豆包 API）
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'local')
# API 向量模型名（EMBEDDING_PROVIDER='api' 时生效）
ARK_EMBEDDING_MODEL = os.getenv('ARK_EMBEDDING_MODEL', 'doubao-embedding-vision-251215')
# 向量维度（不同模型维度不同，请根据实际模型设置）
_default_dim = '2048' if EMBEDDING_PROVIDER == 'api' else '384'
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', _default_dim))

# 分批请求配置
MAX_TABLE_LENGTH_PER_BATCH = int(os.getenv('MAX_TABLE_LENGTH_PER_BATCH', '20000'))
MAX_BATCHES = int(os.getenv('MAX_BATCHES', '5'))
MIN_TABLES_PER_BATCH = int(os.getenv('MIN_TABLES_PER_BATCH', '2'))


# ==================== SQL 清洗与安全 ====================
def extract_final_sql(raw_content: str) -> str:
    """从 CoT 输出中提取 ## 最终SQL 后的 SQL，再交给 clean_sql 清洗"""
    if not raw_content:
        return ""

    match = re.search(r'##\s*最终\s*SQL\s*\n+(.*)', raw_content, re.DOTALL | re.IGNORECASE)
    if match:
        return clean_sql(match.group(1))

    return clean_sql(raw_content)


def clean_sql(raw_content: str) -> str:
    """清理 SQL 语句（去除 markdown 标记、注释，提取有效 SQL）"""
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
            return ""

    return sql


_SQL_DANGEROUS_PATTERNS = [
    r'\bINTO\s+OUTFILE\b',
    r'\bINTO\s+DUMPFILE\b',
    r'\bLOAD_FILE\b',
    r'\bINSERT\s+INTO\b',
    r'\bUPDATE\s+\w+\s+SET\b',
    r'\bDELETE\s+FROM\b',
    r'\bDROP\s+(TABLE|DATABASE|SCHEMA|INDEX)\b',
    r'\bALTER\s+(TABLE|DATABASE)\b',
    r'\bCREATE\s+(TABLE|DATABASE|SCHEMA|INDEX)\b',
    r'\bTRUNCATE\b',
    r'\bGRANT\b',
    r'\bREVOKE\b',
    r'\bEXEC(UTE)?\b',
    r'\bCALL\b',
]


def validate_sql_safety(sql: str) -> None:
    """检查 SQL 是否包含危险操作，不安全则抛出 ValueError"""
    sql_upper = sql.upper()
    for pattern in _SQL_DANGEROUS_PATTERNS:
        if re.search(pattern, sql_upper, re.IGNORECASE):
            raise ValueError(f"SQL 包含不允许的操作: {pattern}")


# ==================== SQL 字段校验（防止 LLM 编造字段） ====================
_SQL_RESERVED = {
    'select', 'from', 'where', 'join', 'left', 'right', 'inner', 'outer', 'full',
    'on', 'and', 'or', 'not', 'as', 'with', 'group', 'by', 'order', 'asc', 'desc',
    'limit', 'offset', 'case', 'when', 'then', 'else', 'end', 'null', 'is', 'in',
    'between', 'like', 'ilike', 'exists', 'distinct', 'union', 'intersect', 'except',
    'all', 'any', 'some', 'having', 'cast', 'true', 'false', 'lateral', 'using',
    'count', 'sum', 'avg', 'min', 'max', 'coalesce', 'nullif', 'greatest', 'least',
    'round', 'floor', 'ceil', 'ceiling', 'abs', 'mod', 'power', 'sqrt', 'sign',
    'length', 'char_length', 'octet_length', 'lower', 'upper', 'initcap',
    'substring', 'substr', 'concat', 'concat_ws', 'trim', 'ltrim', 'rtrim',
    'replace', 'translate', 'split_part', 'regexp_replace', 'regexp_match', 'position',
    'to_char', 'to_date', 'to_timestamp', 'to_number', 'date_trunc', 'date_part', 'extract',
    'now', 'current_date', 'current_time', 'current_timestamp', 'localtime', 'localtimestamp',
    'interval', 'age', 'epoch',
    'numeric', 'decimal', 'int', 'int2', 'int4', 'int8', 'bigint', 'integer', 'smallint',
    'text', 'varchar', 'char', 'character', 'date', 'timestamp', 'timestamptz', 'time',
    'boolean', 'bool', 'real', 'double', 'precision', 'float', 'json', 'jsonb', 'uuid',
    'over', 'partition', 'row_number', 'rank', 'dense_rank', 'lag', 'lead',
    'first_value', 'last_value', 'ntile', 'percent_rank', 'cume_dist',
    'nulls', 'first', 'last', 'array', 'array_agg', 'string_agg', 'unnest',
    'window', 'within', 'rows', 'range', 'preceding', 'following', 'unbounded', 'current',
    'returning', 'collate', 'similar', 'tsvector', 'tsquery',
}


def _extract_schema_identifiers(formatted_tables: str):
    """从 formatted_tables 中提取:
        - table_to_fields: {table_name (lower): set(field_names lower)}
        - valid_tables: set of table names (lower, without schema)
        - valid_schemas: set of schema names (lower)
    """
    table_to_fields = {}
    valid_tables = set()
    valid_schemas = set()
    current_table = None

    for line in formatted_tables.split('\n'):
        m = re.match(r'^表名:\s*([\w.]+)', line)
        if m:
            full = m.group(1).lower()
            if '.' in full:
                schema, table = full.split('.', 1)
                valid_schemas.add(schema)
                valid_tables.add(table)
                current_table = table
            else:
                valid_tables.add(full)
                current_table = full
            table_to_fields.setdefault(current_table, set())
            continue
        m = re.match(r'^\s+(\w+)\s*\[', line)
        if m and current_table is not None:
            table_to_fields[current_table].add(m.group(1).lower())

    return table_to_fields, valid_tables, valid_schemas


_SQL_ALIAS_KEYWORDS = {
    'where', 'on', 'group', 'order', 'having', 'left', 'right', 'inner', 'outer',
    'full', 'cross', 'lateral', 'using', 'limit', 'offset', 'as', 'union',
    'intersect', 'except', 'join', 'with', 'select', 'and', 'or', 'not',
}


def _parse_alias_to_table(sql_clean: str, valid_tables: set):
    """从 SQL 提取 alias/table → real table name 的映射

    覆盖以下形式：
        FROM schema.table alias / FROM schema.table AS alias
        JOIN schema.table alias / JOIN schema.table AS alias
        FROM schema.table  (无别名，表名自身就是引用前缀)
    """
    alias_to_table = {}

    pattern = re.compile(
        r'\b(?:FROM|JOIN)\s+([\w.]+)(?:\s+(?:AS\s+)?(\w+))?',
        re.IGNORECASE,
    )
    for m in pattern.finditer(sql_clean):
        full_table = m.group(1).lower()
        table_name = full_table.split('.')[-1]

        # 表名自身永远是合法引用前缀（如 SELECT bi.foo.x FROM bi.foo）
        if table_name in valid_tables:
            alias_to_table[table_name] = table_name

        alias = m.group(2)
        if alias:
            alias_lower = alias.lower()
            # alias 不能是 SQL 关键字（避免把 "JOIN x ON" 中的 ON 当成 x 的别名）
            if alias_lower not in _SQL_ALIAS_KEYWORDS:
                alias_to_table[alias_lower] = table_name

    return alias_to_table


def find_invalid_sql_identifiers(sql: str, formatted_tables: str) -> List[str]:
    """检查 SQL 中是否引用了不存在于 schema 的字段。

    返回疑似编造的标识符列表（去重、排序）。空列表表示未发现明显问题。

    精准模式：对 `alias.field` 形式，会按"alias→table→fields"链路检查，
    即便字段名在其他表里存在，但**这张表**里没有也会被标记。
    """
    if not sql or not formatted_tables:
        return []

    table_to_fields, valid_tables, valid_schemas = _extract_schema_identifiers(formatted_tables)
    if not table_to_fields:
        return []

    all_valid_fields = set()
    for fields in table_to_fields.values():
        all_valid_fields.update(fields)

    # 去掉字符串字面量与注释，避免误把字符串内容当作标识符
    sql_clean = re.sub(r"'(?:[^']|'')*'", '', sql)
    sql_clean = re.sub(r'"(?:[^"])*"', '', sql_clean)
    sql_clean = re.sub(r'--[^\n]*', '', sql_clean)
    sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)

    cte_names = {m.group(1).lower() for m in re.finditer(r'\b(\w+)\s+AS\s*\(', sql_clean, re.IGNORECASE)}
    aliases_after_as = {m.group(1).lower() for m in re.finditer(r'\bAS\s+(\w+)', sql_clean, re.IGNORECASE)}
    used_left_idents = {m.group(1).lower() for m in re.finditer(r'\b(\w+)\.\w+\b', sql_clean)}
    alias_to_table = _parse_alias_to_table(sql_clean, valid_tables)

    suspicious = set()

    # 1) dotted: alias.field — 优先精准检查 alias 指向的表
    for m in re.finditer(r'\b(\w+)\.(\w+)\b', sql_clean):
        left, right = m.group(1).lower(), m.group(2).lower()
        if right == '*':
            continue
        # schema.table 形式
        if left in valid_schemas and right in valid_tables:
            continue
        # field 别名 / CTE 名
        if right in aliases_after_as or right in cte_names:
            continue
        target_table = alias_to_table.get(left)
        if target_table and target_table in table_to_fields:
            # 精准命中：检查目标表里是否真有这个字段
            if right not in table_to_fields[target_table]:
                suspicious.add(m.group(2))
            continue
        # 没找到表（可能引用了 CTE 列，比如 cte_name.col），降级用全集合
        if left in cte_names:
            continue
        if right not in all_valid_fields:
            suspicious.add(m.group(2))

    # 2) bare identifier — 排除关键字、合法字段/表/schema、CTE 名、别名、表别名
    for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z_0-9]*)\b', sql_clean):
        ident = m.group(1)
        ident_lower = ident.lower()
        if ident_lower in _SQL_RESERVED:
            continue
        if ident_lower in all_valid_fields:
            continue
        if ident_lower in valid_tables or ident_lower in valid_schemas:
            continue
        if ident_lower in cte_names or ident_lower in aliases_after_as:
            continue
        if ident_lower in used_left_idents:
            continue
        suspicious.add(ident)

    return sorted(suspicious)


def extract_relevant_schema_blocks(sql: str, formatted_tables: str) -> str:
    """根据 SQL 中 FROM/JOIN 引用的表，从 formatted_tables 抽出这些表的完整 schema 块。

    用于"分批 → 单批回退"场景：避免把全部 schema 截断到 8000 字符导致信息丢失，
    只保留最相关的几张表的完整字段。
    """
    if not sql or not formatted_tables:
        return formatted_tables

    sql_clean = re.sub(r"'(?:[^']|'')*'", '', sql)
    sql_clean = re.sub(r'--[^\n]*', '', sql_clean)

    referenced = set()
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+([\w.]+)', sql_clean, re.IGNORECASE):
        full = m.group(1).lower()
        referenced.add(full)
        referenced.add(full.split('.')[-1])

    if not referenced:
        return formatted_tables

    blocks = formatted_tables.split('\n\n')
    relevant = []
    for block in blocks:
        m = re.match(r'^表名:\s*([\w.]+)', block.lstrip())
        if not m:
            continue
        full = m.group(1).lower()
        table = full.split('.')[-1]
        if full in referenced or table in referenced:
            relevant.append(block)

    return '\n\n'.join(relevant) if relevant else formatted_tables


def build_table_field_hint(sql: str, formatted_tables: str) -> str:
    if not sql or not formatted_tables:
        return ""

    table_to_fields, valid_tables, _ = _extract_schema_identifiers(formatted_tables)
    if not table_to_fields:
        return ""

    sql_clean = re.sub(r"'(?:[^']|'')*'", '', sql)
    sql_clean = re.sub(r'--[^\n]*', '', sql_clean)

    referenced_tables = []
    seen = set()
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+([\w.]+)', sql_clean, re.IGNORECASE):
        full = m.group(1).lower()
        table = full.split('.')[-1]
        if table in valid_tables and table not in seen:
            seen.add(table)
            referenced_tables.append((full, table))

    if not referenced_tables:
        return ""

    lines = []
    for full, table in referenced_tables:
        fields = sorted(table_to_fields.get(table, set()))
        if fields:
            lines.append(f"- {full} 的真实字段（仅这些可用）: {', '.join(fields)}")
        else:
            lines.append(f"- {full}: 字段未知")
    return '\n'.join(lines)



# ==================== TTL 缓存 ====================
class TTLCache:
    """简单的线程安全 TTL 缓存"""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 32):
        self._cache: OrderedDict = OrderedDict()
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                value, ts = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._cache.move_to_end(key)
                    return value
                del self._cache[key]
        return None

    def set(self, key: str, value):
        with self._lock:
            self._cache[key] = (value, time.time())
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, key: str = None):
        with self._lock:
            if key:
                self._cache.pop(key, None)
            else:
                self._cache.clear()


_schema_cache = TTLCache(ttl_seconds=300)
_vectors_cache = TTLCache(ttl_seconds=3600, max_size=8)


# ==================== 限流器 ====================
class RateLimiter:
    """简单的滑动窗口限流"""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._requests: List[float] = []
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = time.time()
        with self._lock:
            self._requests = [t for t in self._requests if now - t < self._window]
            if len(self._requests) >= self._max:
                return False
            self._requests.append(now)
            return True


_nl_query_limiter = RateLimiter(max_requests=20, window_seconds=60)


# ==================== 重试装饰器 ====================
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """指数退避重试"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            wait = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_attempts:
                        print(f"    ⚠️ 第{attempt}次调用失败，{wait:.1f}s 后重试: {e}")
                        time.sleep(wait)
                        wait *= backoff
            raise last_exc
        return wrapper
    return decorator


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
