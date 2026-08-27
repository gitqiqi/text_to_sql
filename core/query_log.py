# core/query_log.py - 查询运行日志记录
import json
from typing import Dict

from flask import has_request_context, request
from sqlalchemy import text

from .auth import build_log_context, get_current_user
from .db_manager import DatabasePoolManager


_log_table_initialized = False


def _ensure_log_table(engine):
    """表由 DBA 预先创建，这里不再做 DDL"""
    global _log_table_initialized
    _log_table_initialized = True


def _infer_action_type(log_data: Dict) -> str:
    if log_data.get('action_type'):
        return log_data['action_type']
    search_mode = log_data.get('search_mode')
    if search_mode == 'manual_sql':
        return 'manual_sql'
    if search_mode == 'upload_match':
        return 'upload_match'
    return 'nl_query'


def _enrich_request_context(log_data: Dict) -> Dict:
    enriched = dict(log_data)
    if not has_request_context():
        return enriched

    action_type = _infer_action_type(enriched)
    context = build_log_context(
        action_type=action_type,
        request_id=enriched.get('request_id'),
        visibility_scope=enriched.get('visibility_scope'),
    )
    for key, value in context.items():
        if enriched.get(key) in (None, ''):
            enriched[key] = value

    user = get_current_user()
    if user:
        # 保留用户快照，方便后续查日志时不必每次联 app_user。
        for key in ('role_id', 'role_name', 'admin_organ_id', 'organ_name'):
            if enriched.get(key) in (None, '') and user.get(key) is not None:
                enriched[key] = user.get(key)

    if enriched.get('user_agent') in (None, ''):
        enriched['user_agent'] = request.headers.get('User-Agent', '')[:1000]
    return enriched


def insert_query_log(db_name_for_engine: str, log_data: Dict):
    """写入一条查询日志（失败静默，不影响主流程）

    db_name_for_engine: 用哪个数据库连接来写日志（通常和被查询的 db 一致）
    log_data: 日志字段字典
    """
    try:
        log_data = _enrich_request_context(log_data)
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
                    total_duration_ms, prompt_tokens, completion_tokens, total_tokens, llm_calls,
                    admin_id, request_id, session_id, client_ip, user_agent,
                    action_type, visibility_scope
                ) VALUES (
                    :db_name, :nl_query, :schema_filter, :search_mode, :selected_table,
                    :top_k, :matched_tables, :generated_sql, :execute_status, :error_message,
                    :result_rows, :search_duration_ms, :llm_duration_ms, :sql_exec_duration_ms,
                    :total_duration_ms, :prompt_tokens, :completion_tokens, :total_tokens, :llm_calls,
                    :admin_id, :request_id, :session_id, :client_ip, :user_agent,
                    :action_type, :visibility_scope
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
                'admin_id': log_data.get('admin_id'),
                'request_id': log_data.get('request_id'),
                'session_id': log_data.get('session_id'),
                'client_ip': log_data.get('client_ip'),
                'user_agent': log_data.get('user_agent'),
                'action_type': log_data.get('action_type'),
                'visibility_scope': log_data.get('visibility_scope'),
            })
            conn.commit()
    except Exception as e:
        print(f"   ⚠️ 写入 query_log 失败（忽略，不影响主流程）: {e}")
