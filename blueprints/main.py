# blueprints/main.py
from flask import render_template, request, jsonify
import time
import pandas as pd
from . import main_bp
from app_core import (
    DatabaseManager, KnowledgeBase, TextToSQLConverter,
    get_available_databases, monitor_function
)


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
    try:
        kb = KnowledgeBase(db_name)
        tables = kb.get_table_list()
        return jsonify({'status': 'success', 'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/knowledge/status', methods=['GET'])
def knowledge_status():
    """获取知识库状态"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    
    kb = KnowledgeBase(db_name)
    knowledge = kb.get_sql_knowledge()
    
    return jsonify({
        'status': 'success',
        'db_name': db_name,
        'row_count': len(knowledge),
        'from_upload': False
    })


@main_bp.route('/execute_sql', methods=['POST'])
def execute_sql():
    """执行SQL查询"""
    try:
        data = request.get_json()
        sql = data.get('sql')
        db_name = data.get('db_name')
        if not sql or not db_name:
            return jsonify({'error': 'Missing sql or db_name'}), 400
        
        db = DatabaseManager(db_name)
        result = db.execute_sql(sql)
        return jsonify({'sql_result': result.to_dict(orient='records'), 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/execute_nl_query', methods=['POST'])
def handle_nl_query():
    """处理自然语言查询"""
    try:
        data = request.get_json()
        if not data or 'nl_query' not in data:
            return jsonify({'error': 'Missing nl_query'}), 400
        
        db_name = data.get('db_name')
        if not db_name:
            return jsonify({'error': 'Missing db_name'}), 400
        
        selected_table = data.get('selected_table')
        if selected_table and isinstance(selected_table, str):
            selected_table = selected_table.strip() or None
        else:
            selected_table = None
        
        use_vector_search = data.get('use_vector_search', True)
        top_k_tables = data.get('top_k_tables', 10)
        
        print(f"\n{'='*60}")
        print(f"📨 查询: {db_name} - {data['nl_query'][:100]}...")
        print(f"   向量检索: {use_vector_search}, 指定表: {selected_table}")
        print(f"{'='*60}")
        
        converter = TextToSQLConverter(db_name)
        
        start_time = time.time()
        sql, result = converter.execute_nl_query(
            nl_query=data['nl_query'],
            selected_table=selected_table,
            use_vector_search=use_vector_search,
            top_k_tables=top_k_tables
        )
        
        elapsed = (time.time() - start_time) * 1000
        print(f"✅ 查询完成，总耗时: {elapsed:.2f} ms")
        
        return jsonify({
            'generated_sql': sql,
            'sql_result': result.to_dict(orient='records'),
            'status': 'success'
        })
    except Exception as e:
        print(f"❌ 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500