# core 包：拆分自原 app_core.py
from .utils import (
    SENTENCE_TRANSFORMER_MODEL,
    EMBEDDING_PROVIDER,
    ARK_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    MAX_TABLE_LENGTH_PER_BATCH,
    MAX_BATCHES,
    MIN_TABLES_PER_BATCH,
    POOL_SIZE,
    MAX_OVERFLOW,
    POOL_PRE_PING,
    TTLCache,
    RateLimiter,
    retry,
    monitor_function,
    clean_sql,
    extract_final_sql,
    validate_sql_safety,
    _schema_cache,
    _nl_query_limiter,
)
from .db_manager import DatabasePoolManager, DatabaseManager
from .llm_client import DouBaoClient
from .knowledge import KnowledgeBase, start_vector_monitor
from .book_code import (
    BookKnowledgeStore,
    sync_book_from_source,
    start_book_sync_monitor,
    is_book_sync_monitor_running,
    get_last_book_sync_result,
)
from .dataworks import DataWorksKnowledgeStore
from .query_log import insert_query_log
from .auth import (
    authenticate_user,
    bootstrap_admin_user,
    build_log_context,
    get_current_user,
    login_user,
    logout_user,
    public_user_payload,
    user_can_view_all,
)
from .user_sync import (
    sync_app_users_from_source,
    start_user_sync_monitor,
    is_user_sync_monitor_running,
    get_last_user_sync_result,
)
from .repos import SQLKnowledgeRepo, GlossaryRepo
from .vector_search import TableSchemaSearcher
from .converter import TextToSQLConverter, precompute_all_embeddings
from .embedding_client import get_embedding_model

__all__ = [
    'SENTENCE_TRANSFORMER_MODEL',
    'EMBEDDING_PROVIDER',
    'ARK_EMBEDDING_MODEL',
    'EMBEDDING_DIM',
    'MAX_TABLE_LENGTH_PER_BATCH',
    'MAX_BATCHES',
    'MIN_TABLES_PER_BATCH',
    'POOL_SIZE',
    'MAX_OVERFLOW',
    'POOL_PRE_PING',
    'TTLCache',
    'RateLimiter',
    'retry',
    'monitor_function',
    'clean_sql',
    'extract_final_sql',
    'validate_sql_safety',
    '_schema_cache',
    '_nl_query_limiter',
    'DatabasePoolManager',
    'DatabaseManager',
    'DouBaoClient',
    'KnowledgeBase',
    'start_vector_monitor',
    'BookKnowledgeStore',
    'sync_book_from_source',
    'start_book_sync_monitor',
    'is_book_sync_monitor_running',
    'get_last_book_sync_result',
    'DataWorksKnowledgeStore',
    'insert_query_log',
    'authenticate_user',
    'bootstrap_admin_user',
    'build_log_context',
    'get_current_user',
    'login_user',
    'logout_user',
    'public_user_payload',
    'user_can_view_all',
    'sync_app_users_from_source',
    'start_user_sync_monitor',
    'is_user_sync_monitor_running',
    'get_last_user_sync_result',
    'SQLKnowledgeRepo',
    'GlossaryRepo',
    'TableSchemaSearcher',
    'TextToSQLConverter',
    'precompute_all_embeddings',
    'get_embedding_model',
]
