# blueprints/__init__.py
from flask import Blueprint

# 创建蓝图实例
main_bp = Blueprint('main', __name__, url_prefix='/')
knowledge_bp = Blueprint('knowledge', __name__, url_prefix='/knowledge')

# 导入路由定义（这一行是关键！）
from . import main
from . import knowledge