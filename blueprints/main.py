# blueprints/main.py
import os
import math
import time
import logging
from datetime import datetime, timedelta
from flask import render_template, request, jsonify
import pandas as pd
from sqlalchemy import text
from werkzeug.utils import secure_filename
from . import main_bp
from config import get_available_databases
from core import (

aisql_logger = logging.getLogger("aisql")
    DatabaseManager, KnowledgeBase, SQLKnowledgeRepo, TextToSQLConverter,
    monitor_function, _nl_query_limiter, insert_query_log,
)
from core.cancellation import registry as cancel_registry, CancelledError
from core.db_manager import DatabasePoolManager
import numpy as np


def safe_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame.to_dict(orient='records') 的安全版本，将 NaN/NaT/Inf 替换为 None"""
    df = df.astype(object).where(df.notna(), None)
    for col in df.columns:
        for i, val in enumerate(df[col]):
            if isinstance(val, float) and (math.isinf(val) or math.isnan(val)):
                df.at[i, col] = None
    return df.to_dict(orient='records')


@main_bp.route('/')
def index():
    """主页"""
    return render_template('index.html')


@main_bp.route('/get_databases', methods=['GET'])
def get_databases():
    """获取可用数据库列表"""
    try:
        databases = get_available_databases()
        return jsonify({'databases': databases, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/db_tables', methods=['GET'])
def get_db_tables():
    """获取数据库表列表"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    schema_filter = request.args.get('schema_name', '').strip() or None
    try:
        kb = KnowledgeBase(db_name)
        tables = kb.get_table_list()
        if schema_filter:
            schemas = set(s.strip() for s in schema_filter.split(',') if s.strip())
            tables = [t for t in tables if t.get('schema') in schemas]
        return jsonify({'status': 'success', 'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/db_schemas', methods=['GET'])
def get_db_schemas():
    """获取数据库的 schema 列表"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    try:
        kb = KnowledgeBase(db_name)
        tables = kb.get_table_list()
        schemas = sorted(set(t['schema'] for t in tables if t.get('schema')))
        return jsonify({'status': 'success', 'schemas': schemas})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/knowledge/status', methods=['GET'])
def knowledge_status():
    """获取知识库状态"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    
    kb = SQLKnowledgeRepo(db_name)
    knowledge = kb.list()
    
    return jsonify({
        'status': 'success',
        'db_name': db_name,
        'row_count': len(knowledge),
        'from_upload': False
    })


@main_bp.route('/execute_sql', methods=['POST'])
def execute_sql():
    """执行SQL查询"""
    start_time = time.time()
    try:
        data = request.get_json()
        sql = data.get('sql')
        db_name = data.get('db_name')
        if not sql or not db_name:
            return jsonify({'error': 'Missing sql or db_name'}), 400
        
        db = DatabaseManager(db_name)
        result = db.execute_sql(sql)
        records = safe_records(result)
        total_duration_ms = (time.time() - start_time) * 1000

        insert_query_log(db_name, {
            'db_name': db_name,
            'nl_query': sql,
            'schema_filter': None,
            'search_mode': 'manual_sql',
            'selected_table': None,
            'top_k': None,
            'matched_tables': None,
            'generated_sql': sql,
            'execute_status': 'success',
            'error_message': None,
            'result_rows': len(result),
            'search_duration_ms': 0,
            'llm_duration_ms': 0,
            'sql_exec_duration_ms': total_duration_ms,
            'total_duration_ms': total_duration_ms,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'llm_calls': 0,
        })

        return jsonify({'sql_result': records, 'status': 'success'})
    except Exception as e:
        total_duration_ms = (time.time() - start_time) * 1000
        insert_query_log(db_name, {
                'db_name': db_name,
                'nl_query': sql,
                'schema_filter': None,
                'search_mode': 'manual_sql',
                'selected_table': None,
                'top_k': None,
                'matched_tables': None,
                'generated_sql': sql,
                'execute_status': 'failed',
                'error_message': str(e)[:1000],
                'result_rows': 0,
                'search_duration_ms': 0,
                'llm_duration_ms': 0,
                'sql_exec_duration_ms': total_duration_ms,
                'total_duration_ms': total_duration_ms,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'llm_calls': 0,
            })
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/query_history', methods=['POST'])
def query_history():
    """获取查询历史记录（支持时间范围筛选）"""
    try:
        data = request.get_json() or {}
        db_name = data.get('db_name')
        limit = min(int(data.get('limit', 50)), 200)
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        search_mode = data.get('search_mode')
        if not db_name:
            return jsonify({'error': 'missing db_name'}), 400

        engine = DatabasePoolManager.get_engine(db_name)

        # 动态构建 WHERE 条件
        where_clauses = ['db_name = :db_name']
        params = {'db_name': db_name, 'limit': limit}

        if start_time:
            where_clauses.append('created_at >= :start_time')
            params['start_time'] = start_time
        if end_time:
            end_dt = datetime.strptime(end_time, '%Y-%m-%d') + timedelta(days=1)
            where_clauses.append('created_at < :end_time_plus')
            params['end_time_plus'] = end_dt.strftime('%Y-%m-%d')
        if search_mode:
            where_clauses.append('search_mode = :search_mode')
            params['search_mode'] = search_mode

        where_sql = ' AND '.join(where_clauses)

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT nl_query, search_mode, generated_sql, execute_status,
                       result_rows, total_duration_ms, error_message, created_at
                FROM knowledge.query_log
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT :limit
            """), params).fetchall()

        history = []
        for row in rows:
            history.append({
                'nl_query': row._mapping.get('nl_query'),
                'search_mode': row._mapping.get('search_mode'),
                'generated_sql': row._mapping.get('generated_sql'),
                'execute_status': row._mapping.get('execute_status'),
                'result_rows': row._mapping.get('result_rows'),
                'total_duration_ms': row._mapping.get('total_duration_ms'),
                'error_message': row._mapping.get('error_message'),
                'created_at': str(row._mapping.get('created_at')) if row._mapping.get('created_at') else None,
            })
        return jsonify({'history': history, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/execute_nl_query', methods=['POST'])
def handle_nl_query():
    """处理自然语言查询"""
    data = request.get_json() or {}
    # 优先使用前端传的 request_id（用于取消），否则后端生成
    client_request_id = (data.get('request_id') or '').strip() if isinstance(data.get('request_id'), str) else ''
    if client_request_id:
        token = cancel_registry.register_with_id(client_request_id)
        request_id, cancel_token = client_request_id, token
    else:
        request_id, cancel_token = cancel_registry.create()

    try:
        # 限流检查
        if not _nl_query_limiter.allow():
            return jsonify({'error': '请求过于频繁，请稍后再试', 'status': 'error'}), 429

        if 'nl_query' not in data:
            return jsonify({'error': 'Missing nl_query'}), 400

        db_name = data.get('db_name')
        if not db_name:
            return jsonify({'error': 'Missing db_name'}), 400

        selected_table = data.get('selected_table')
        if selected_table and isinstance(selected_table, str):
            selected_table = selected_table.strip() or None
        else:
            selected_table = None

        schema_name = data.get('schema_name', '').strip() or None
        use_vector_search = data.get('use_vector_search', True)
        top_k_tables = data.get('top_k_tables', 10)
        embedding_provider = data.get('embedding_provider', '').strip() or None

        aisql_logger.info("=" * 60)
        aisql_logger.info("查询 | db=%s | request_id=%s", db_name, request_id)
        aisql_logger.info("用户问题: %s", data['nl_query'][:200])
        aisql_logger.info("向量检索=%s | 指定表=%s | schema=%s | 向量模型=%s",
                          use_vector_search, selected_table, schema_name,
                          embedding_provider or '默认')

        converter = TextToSQLConverter(db_name)

        start_time = time.time()
        sql, result = converter.execute_nl_query(
            nl_query=data['nl_query'],
            selected_table=selected_table,
            use_vector_search=use_vector_search,
            top_k_tables=top_k_tables,
            schema_filter=schema_name,
            cancel_token=cancel_token,
            embedding_provider=embedding_provider,
        )

        elapsed = (time.time() - start_time) * 1000
        aisql_logger.info("完成 | 耗时=%.0fms | 行数=%s | SQL: %s",
                          elapsed, len(result), sql[:300])

        MAX_DISPLAY_ROWS = 500
        total_rows = len(result)
        if total_rows > MAX_DISPLAY_ROWS:
            display_result = safe_records(result.head(MAX_DISPLAY_ROWS))
        else:
            display_result = safe_records(result)

        return jsonify({
            'request_id': request_id,
            'generated_sql': sql,
            'sql_result': display_result,
            'total_rows': total_rows,
            'status': 'success'
        })
    except CancelledError as e:
        aisql_logger.warning("取消 | request_id=%s", request_id)
        return jsonify({'error': '查询已取消', 'status': 'cancelled', 'request_id': request_id}), 499
    except Exception as e:
        aisql_logger.error("失败 | request_id=%s | error=%s", request_id, str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error', 'request_id': request_id}), 500
    finally:
        cancel_registry.cleanup(request_id)


@main_bp.route('/api/rebuild_all_vectors', methods=['POST'])
def rebuild_all_vectors():
    """手动触发全量向量重建"""
    data = request.get_json() or {}
    db_name = data.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    try:
        from core.knowledge import KnowledgeBase
        kb = KnowledgeBase(db_name)
        table_records, vector_texts = kb.get_vector_texts()
        if not table_records:
            return jsonify({'status': 'success', 'message': '无表需更新'})
        for provider in ('local', 'api'):
            from core.embedding_client import get_embedding_model
            model = get_embedding_model(provider)
            kb.save_embeddings_incrementally(model, table_records, vector_texts)
        kb.rebuild_knowledge_vectors()
        return jsonify({'status': 'success', 'message': f'向量更新完成 ({db_name})'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/cancel_query', methods=['POST'])
def cancel_query():
    """取消正在执行的查询"""
    data = request.get_json() or {}
    request_id = data.get('request_id')
    if not request_id:
        return jsonify({'error': 'Missing request_id', 'status': 'error'}), 400
    found = cancel_registry.cancel(request_id)
    if found:
        print(f"⛔ 收到取消请求: {request_id}")
        return jsonify({'status': 'cancelled', 'request_id': request_id})
    return jsonify({'status': 'not_found', 'request_id': request_id}), 404


@main_bp.route('/api/upload_excel', methods=['POST'])
def upload_excel():
    """上传 Excel 并导入到 tmp schema 下的指定表"""
    db_name = request.form.get('db_name', '').strip()
    table_name = request.form.get('table_name', '').strip()
    file = request.files.get('file')

    if not db_name:
        return jsonify({'error': '请选择数据库', 'status': 'error'}), 400
    if not table_name:
        return jsonify({'error': '请输入表名', 'status': 'error'}), 400
    if not file or not file.filename:
        return jsonify({'error': '请选择文件', 'status': 'error'}), 400
    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        return jsonify({'error': '仅支持 .xlsx 或 .xls 格式', 'status': 'error'}), 400

    safe_name = table_name.replace(' ', '_').replace('-', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
    if not safe_name:
        return jsonify({'error': '表名不合法，请使用字母、数字、下划线', 'status': 'error'}), 400

    try:
        df = pd.read_excel(file)
        if df.empty:
            return jsonify({'error': 'Excel 文件为空', 'status': 'error'}), 400

        df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]

        engine = DatabasePoolManager.get_engine(db_name)

        # Phase 1: DDL - DROP + CREATE 在独立事务中
        type_map = {
            'int64': 'BIGINT', 'Int64': 'BIGINT',
            'float64': 'DOUBLE PRECISION', 'Float64': 'DOUBLE PRECISION',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'object': 'TEXT',
        }
        cols = []
        for name, dtype in df.dtypes.items():
            pg_type = type_map.get(str(dtype), 'TEXT')
            safe_col = str(name).replace('"', '""')
            cols.append(f'"{safe_col}" {pg_type}')
        with engine.connect() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS tmp."{safe_name}" CASCADE'))
            conn.execute(text(f'CREATE TABLE tmp."{safe_name}" ({", ".join(cols)})'))
            conn.commit()

        # Phase 2: DML - INSERT 在独立事务中
        df.to_sql(safe_name, engine, schema='tmp', if_exists='append', index=False, method=None)

        row_count = len(df)
        full_name = f"tmp.{safe_name}"
        return jsonify({
            'status': 'success',
            'table_name': full_name,
            'row_count': row_count,
            'columns': list(df.columns),
            'message': f'✅ 成功导入 {row_count} 行数据到 {full_name}'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500