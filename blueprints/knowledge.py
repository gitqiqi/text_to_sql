# blueprints/knowledge.py
from flask import render_template, request, jsonify
from . import knowledge_bp
from app_core import KnowledgeBase


@knowledge_bp.route('/management')
def knowledge_management():
    """知识库管理页面"""
    return render_template('knowledge_management.html')


@knowledge_bp.route('/api/list', methods=['GET'])
def get_knowledge_list():
    """获取知识库列表"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    
    try:
        kb = KnowledgeBase(db_name)
        knowledge = kb.get_sql_knowledge()
        return jsonify({'knowledge': knowledge, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@knowledge_bp.route('/api/add', methods=['POST'])
def add_knowledge():
    """添加知识条目"""
    data = request.get_json()
    db_name = data.get('db_name', '').strip()
    question = data.get('question', '').strip()
    sql = data.get('sql', '').strip()
    
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    if not sql:
        return jsonify({'error': 'SQL不能为空'}), 400
    
    try:
        kb = KnowledgeBase(db_name)
        result = kb.add_knowledge(question, sql)
        return jsonify({'status': 'success', 'id': result['id']})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@knowledge_bp.route('/api/update/<int:knowledge_id>', methods=['PUT'])
def update_knowledge(knowledge_id):
    """更新知识条目"""
    data = request.get_json()
    db_name = data.get('db_name', '').strip()
    question = data.get('question', '').strip()
    sql = data.get('sql', '').strip()
    
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    
    try:
        kb = KnowledgeBase(db_name)
        success = kb.update_knowledge(knowledge_id, question, sql)
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': '知识条目不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@knowledge_bp.route('/api/delete/<int:knowledge_id>', methods=['DELETE'])
def delete_knowledge(knowledge_id):
    """删除知识条目"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    
    try:
        kb = KnowledgeBase(db_name)
        success = kb.delete_knowledge(knowledge_id)
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': '知识条目不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@knowledge_bp.route('/api/status', methods=['GET'])
def get_status():
    """获取知识库状态"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    
    try:
        kb = KnowledgeBase(db_name)
        knowledge = kb.get_sql_knowledge()
        return jsonify({
            'status': 'success',
            'db_name': db_name,
            'row_count': len(knowledge)
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500