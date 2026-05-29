# core/query_log.py - 查询运行日志记录
import json
from typing import Dict

from sqlalchemy import text

from .db_manager import DatabasePoolManager


_log_table_initialized = False


def _ensure_log_table(engine):
    """表由 DBA 预先创建，这里不再做 DDL"""
    global _log_table_initialized
    _log_table_initialized = True


def insert_query_log(db_name_for_engine: str, log_data: Dict):
    """写入一条查询日志（失败静默，不影响主流程）

    db_name_for_engine: 用哪个数据库连接来写日志（通常和被查询的 db 一致）
    log_data: 日志字段字典
    """
    try:
        engine = DatabasePoolManager.get_engine(db_name_for_engine)
        _ensure_log_table(engine)

        # 把 list/dict 字段转 JSON 字符串
        if isinstance(log_data.get('matched_tables'), (list, dict)):
            log_data['matched_tables'] = json.dumps(log_data['matched_tables'], ensure_ascii=False)

        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO knowledge.query_log (
                    db_name, nl_query, schema_filter, search_mode, selected_table,
                    top_k, matched_tables, generated_sql, execute_status, error_message,
                    result_rows, search_duration_ms, llm_duration_ms, sql_exec_duration_ms,
                    total_duration_ms, prompt_tokens, completion_tokens, total_tokens, llm_calls
                ) VALUES (
                    :db_name, :nl_query, :schema_filter, :search_mode, :selected_table,
                    :top_k, :matched_tables, :generated_sql, :execute_status, :error_message,
                    :result_rows, :search_duration_ms, :llm_duration_ms, :sql_exec_duration_ms,
                    :total_duration_ms, :prompt_tokens, :completion_tokens, :total_tokens, :llm_calls
                )
            """), {
                'db_name': log_data.get('db_name'),
                'nl_query': log_data.get('nl_query'),
                'schema_filter': log_data.get('schema_filter'),
                'search_mode': log_data.get('search_mode'),
                'selected_table': log_data.get('selected_table'),
                'top_k': log_data.get('top_k'),
                'matched_tables': log_data.get('matched_tables'),
                'generated_sql': log_data.get('generated_sql'),
                'execute_status': log_data.get('execute_status'),
                'error_message': log_data.get('error_message'),
                'result_rows': log_data.get('result_rows'),
                'search_duration_ms': log_data.get('search_duration_ms'),
                'llm_duration_ms': log_data.get('llm_duration_ms'),
                'sql_exec_duration_ms': log_data.get('sql_exec_duration_ms'),
                'total_duration_ms': log_data.get('total_duration_ms'),
                'prompt_tokens': log_data.get('prompt_tokens'),
                'completion_tokens': log_data.get('completion_tokens'),
                'total_tokens': log_data.get('total_tokens'),
                'llm_calls': log_data.get('llm_calls'),
            })
            conn.commit()
    except Exception as e:
        print(f"   ⚠️ 写入 query_log 失败（忽略，不影响主流程）: {e}")
