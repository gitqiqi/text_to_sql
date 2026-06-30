import logging

from flask import Flask
from blueprints import main_bp, knowledge_bp
from core.knowledge import start_vector_monitor
from config import get_available_databases, EXCLUDED_DATABASES

from core.log_setup import setup_logging

app = Flask(__name__)
setup_logging(app)

logger = logging.getLogger(__name__)

# 注册蓝图
app.register_blueprint(main_bp)
app.register_blueprint(knowledge_bp)

if __name__ == '__main__':
    logger.info("数据库: PostgreSQL/Hologres, SQLite")
    logger.info("知识库存储: PostgreSQL数据库 (knowledge.db_knowledge)")

    # 启动向量监控线程
    db_configs = get_available_databases()
    db_names = [db['id'] for db in db_configs if db['id'] not in EXCLUDED_DATABASES]
    start_vector_monitor(db_names)

    logger.info("服务启动: http://localhost:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)