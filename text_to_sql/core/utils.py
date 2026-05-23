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

# 分批请求配置
MAX_TABLE_LENGTH_PER_BATCH = int(os.getenv('MAX_TABLE_LENGTH_PER_BATCH', '8000'))
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
