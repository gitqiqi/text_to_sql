# 数据库配置文件
# 作者: Jamesenh

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent
KB_META_PATH = _PROJECT_ROOT / "knowledge_meta.json"


def _load_kb_meta() -> Dict[str, Any]:
    if not KB_META_PATH.is_file():
        return {}
    try:
        return json.loads(KB_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_kb_meta(meta: Dict[str, Any]) -> None:
    KB_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def set_knowledge_file_override(db_name: str, file_path: str, sheet_name: Optional[str]) -> None:
    """记录通过页面上传替换后的知识库 Excel 路径与工作表名。"""
    meta = _load_kb_meta()
    meta[db_name] = {"file_path": file_path, "sheet_name": sheet_name}
    _write_kb_meta(meta)


def clear_knowledge_file_override(db_name: str) -> None:
    """恢复为 KNOWLEDGE_BASE_CONFIGS 中的默认路径。"""
    meta = _load_kb_meta()
    meta.pop(db_name, None)
    _write_kb_meta(meta)


def get_knowledge_override(db_name: str) -> Optional[Dict[str, Any]]:
    return _load_kb_meta().get(db_name)

# 数据库配置字典
DATABASE_CONFIGS = {
     'hologres': {
        'type': 'postgresql',
        'host': os.getenv('DB_HOLOGRES_HOST'),
        'port': os.getenv('DB_HOLOGRES_PORT', '80'),
        'name': os.getenv('DB_HOLOGRES_DATABASE'),
        'user': os.getenv('DB_HOLOGRES_USER'),
        'password': os.getenv('DB_HOLOGRES_PASSWORD'),
        'sslmode': os.getenv('DB_HOLOGRES_SSLMODE', 'prefer'),
        'schema': os.getenv('DB_HOLOGRES_SCHEMA', 'bi'),
        'display_name': 'Hologres',
        'description': '阿里云 Hologres 实时数仓'
    },
    # 盒子数据源
    'book': {
        'type': os.getenv('DB_BOX_TYPE', 'mysql'),
        'host': os.getenv('DB_BOX_HOST'),
        'port': os.getenv('DB_BOX_PORT', '80'),
        'name': os.getenv('DB_BOX_NAME'),
        'user': os.getenv('DB_BOX_USER'),
        'password': os.getenv('DB_BOX_PASSWORD'),
        'display_name': '盒子数据源',
        'description': '盒子系统数据库，包含图书、课程等信息'
    },
    
    # 火苗会议
    'huomiao': {
        'type': os.getenv('DB_HUOMIAO_TYPE', 'mysql'),
        'host': os.getenv('DB_HUOMIAO_HOST'),
        'port': os.getenv('DB_HUOMIAO_PORT', '3306'),
        'name': os.getenv('DB_HUOMIAO_NAME'),
        'user': os.getenv('DB_HUOMIAO_USER'),
        'password': os.getenv('DB_HUOMIAO_PASSWORD'),
        'display_name': '火苗会议',
        'description': '火苗会议系统数据库，包含会议、用户等信息'
    },
    
    # 根源优课
    'uk': {
        'type': os.getenv('DB_UK_TYPE', 'mysql'),
        'host': os.getenv('DB_UK_HOST'),
        'port': os.getenv('DB_UK_PORT', '3306'),
        'name': os.getenv('DB_UK_NAME'),
        'user': os.getenv('DB_UK_USER'),
        'password': os.getenv('DB_UK_PASSWORD'),
        'display_name': '根源优课',
        'description': '根源优课系统数据库，包含课程、学习记录等信息'
    },
    
    # 本地日志数据库
    'local_log': {
        'type': 'mysql',
        'host': os.getenv('DB_LOCAL_HOST', 'localhost'),
        'port': os.getenv('DB_LOCAL_PORT', '3306'),
        'name': os.getenv('DB_LOCAL_NAME', 'text_to_sql_logs'),
        'user': os.getenv('DB_LOCAL_USER', 'root'),
        'password': os.getenv('DB_LOCAL_PASSWORD', ''),
        'display_name': '本地日志数据库',
        'description': '本地MySQL数据库，用于存储查询日志'
    }

    # 阿里云 Hologres（PostgreSQL 协议），在项目根 .env 中配置：
    # DB_HOLOGRES_HOST / DB_HOLOGRES_PORT（默认 80）/ DB_HOLOGRES_USER / DB_HOLOGRES_PASSWORD
    # 库名：DB_HOLOGRES_NAME 或 DB_HOLOGRES_DATABASE（二选一即可）
    # DB_HOLOGRES_SSLMODE（可选，默认 prefer）/ DB_HOLOGRES_SCHEMA（默认 bi，供业务使用）
}

# 知识库文件配置
KNOWLEDGE_BASE_CONFIGS = {
    'hologres': {
        'file_path': '/Users/cherry/Desktop/数据库表结构/holo.xlsx',
        'sheet_name': '知识库'
    },
    'book': {
        'file_path': '/Users/cherry/Desktop/数据库表结构/盒子-book.xlsx',
        'sheet_name': '知识库'
    },
    'huomiao': {
        'file_path': '/Users/cherry/Desktop/数据库表结构/火苗-uclass.xlsx',
        'sheet_name': '知识库'
    },
    'uk': {
        'file_path': '/Users/cherry/Desktop/数据库表结构/根源优课uclass.xlsx',
        'sheet_name': '知识库'
    }
}

def get_database_config(db_name: str):
    """获取指定数据库的配置"""
    return DATABASE_CONFIGS.get(db_name)

def get_available_databases():
    """获取所有可用数据库的列表"""
    return [
        {
            'id': db_id,
            'name': config['display_name'],
            'description': config['description']
        }
        for db_id, config in DATABASE_CONFIGS.items()
    ]

def get_knowledge_base_config(db_name: str):
    """获取指定数据库的知识库配置（若存在上传覆盖则使用覆盖文件）。"""
    cfg = KNOWLEDGE_BASE_CONFIGS.get(db_name, KNOWLEDGE_BASE_CONFIGS['book'])
    if db_name == 'db' and not (cfg.get('file_path') or '').strip():
        cfg = KNOWLEDGE_BASE_CONFIGS['book']
    ov = _load_kb_meta().get(db_name)
    if ov and ov.get('file_path') and os.path.isfile(ov['file_path']):
        merged = dict(cfg)
        merged['file_path'] = ov['file_path']
        if ov.get('sheet_name'):
            merged['sheet_name'] = str(ov['sheet_name'])
        return merged
    return cfg 