# app.py - 主入口
from flask import Flask
from blueprints import main_bp, knowledge_bp

app = Flask(__name__)

# 注册蓝图
app.register_blueprint(main_bp)
app.register_blueprint(knowledge_bp)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Text2SQL 应用启动")
    print("   数据库: PostgreSQL/Hologres, SQLite")
    print("   知识库存储: PostgreSQL数据库 (knowledge.db_knowledge)")
    print("="*60)
    
    print("\n" + "="*60)
    print("✅ 启动完成")
    print("   - 主页: http://localhost:5000/")
    print("   - 知识库管理: http://localhost:5000/knowledge/management")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)