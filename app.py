# app.py - 主入口
import os
from datetime import timedelta

from flask import Flask, jsonify, redirect, request, url_for
from blueprints import main_bp, knowledge_bp
from core.auth import bootstrap_admin_user, get_current_user
from core.book_code import start_book_sync_monitor
from core.knowledge import start_vector_monitor
from core.user_sync import start_user_sync_monitor
from config import get_available_databases, EXCLUDED_DATABASES

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.getenv("SECRET_KEY", "text2sql-dev-secret-change-me"))
app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(hours=int(os.getenv("APP_SESSION_HOURS", "12"))),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

_auth_bootstrap_checked = False
_auth_user_sync_checked = False
_book_sync_checked = False


def _wants_json_response() -> bool:
    if (
        request.path.startswith('/api/')
        or request.path.startswith('/knowledge/api/')
        or request.path.startswith('/execute_')
        or request.path.startswith('/cancel_')
    ):
        return True
    return 'application/json' in request.headers.get('Accept', '')


@app.before_request
def require_login():
    """除登录接口和静态资源外，所有页面/API 都要求登录。"""
    global _auth_bootstrap_checked, _auth_user_sync_checked, _book_sync_checked
    if not _auth_bootstrap_checked:
        bootstrap_ok = bootstrap_admin_user()
        has_bootstrap_config = bool(os.getenv("APP_ADMIN_PASSWORD") or os.getenv("APP_ADMIN_PASSWORD_HASH"))
        _auth_bootstrap_checked = bootstrap_ok or not has_bootstrap_config

    if not _auth_user_sync_checked:
        start_user_sync_monitor()
        _auth_user_sync_checked = True

    if not _book_sync_checked:
        start_book_sync_monitor()
        _book_sync_checked = True

    if request.method == 'OPTIONS':
        return None

    public_endpoints = {
        'static',
        'main.login_page',
        'main.login',
        'main.current_user',
        'main.logout',
    }
    if request.endpoint in public_endpoints:
        return None

    if get_current_user(silent=True):
        return None

    if _wants_json_response():
        return jsonify({'status': 'unauthorized', 'error': '请先登录'}), 401

    return redirect(url_for('main.login_page', next=request.full_path))


@app.after_request
def add_local_preview_cors_headers(response):
    """Allow direct file:// preview pages to call the local Flask API."""
    if request.headers.get('Origin') == 'null':
        response.headers['Access-Control-Allow-Origin'] = 'null'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PATCH,DELETE,OPTIONS'
    return response

# 注册蓝图
app.register_blueprint(main_bp)
app.register_blueprint(knowledge_bp)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Text2SQL 应用启动")
    print("   数据库: PostgreSQL/Hologres, SQLite")
    print("   知识库存储: PostgreSQL数据库 (knowledge.db_knowledge)")
    print("="*60)

    # 先独立写入 admin 引导账号，再启动各类后台线程。
    bootstrap_ok = bootstrap_admin_user()
    has_bootstrap_config = bool(os.getenv("APP_ADMIN_PASSWORD") or os.getenv("APP_ADMIN_PASSWORD_HASH"))
    _auth_bootstrap_checked = bootstrap_ok or not has_bootstrap_config

    # 启动向量监控线程。debug reloader 会产生父子进程，只允许真正服务进程启动监控。
    db_configs = get_available_databases()
    db_names = [db['id'] for db in db_configs if db['id'] not in EXCLUDED_DATABASES]
    if os.getenv('WERKZEUG_RUN_MAIN') in (None, 'true'):
        start_vector_monitor(db_names)
        start_user_sync_monitor()
        start_book_sync_monitor()
        _auth_user_sync_checked = True
        _book_sync_checked = True


    print("\n" + "="*60)
    print("✅ 启动完成")
    print("   - 主页: http://localhost:5000/")
    print("   - 知识库管理: http://localhost:5000/knowledge/management")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
