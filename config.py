# config.py - 精简版
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# 不参与向量构建的数据库
EXCLUDED_DATABASES: List[str] = [
    'hg_recyclebin',
]

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent
KB_META_PATH = _PROJECT_ROOT / "knowledge_meta.json"


def _load_kb_meta() -> Dict:
    """加载知识库覆盖配置"""
    if not KB_META_PATH.is_file():
        return {}
    try:
        return json.loads(KB_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_kb_meta(meta: Dict) -> None:
    """保存知识库覆盖配置"""
    KB_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def set_knowledge_file_override(db_name: str, file_path: str, sheet_name: Optional[str]) -> None:
    """记录通过页面上传替换后的知识库 Excel 路径"""
    meta = _load_kb_meta()
    meta[db_name] = {"file_path": file_path, "sheet_name": sheet_name}
    _write_kb_meta(meta)


def clear_knowledge_file_override(db_name: str) -> None:
    """恢复为默认知识库路径"""
    meta = _load_kb_meta()
    meta.pop(db_name, None)
    _write_kb_meta(meta)


def get_knowledge_override(db_name: str) -> Optional[Dict]:
    """获取知识库覆盖配置"""
    return _load_kb_meta().get(db_name)


# ==================== 数据库配置（仅保留PostgreSQL和SQLite） ====================
DATABASE_CONFIGS = {
    'hologres': {
        'type': 'postgresql',
        'host': os.getenv('DB_HOLOGRES_HOST'),
        'port': os.getenv('DB_HOLOGRES_PORT', '80'),
        'name': os.getenv('DB_HOLOGRES_DATABASE'),
        'user': os.getenv('DB_HOLOGRES_USER'),
        'password': os.getenv('DB_HOLOGRES_PASSWORD'),
        'sslmode': os.getenv('DB_HOLOGRES_SSLMODE', 'prefer'),
        'display_name': 'Hologres',
        'description': '阿里云 Hologres 实时数仓'
    },
}

# 可根据需要添加SQLite测试数据库
# 'test': {
#     'type': 'sqlite',
#     'file_path': '/path/to/test.db',
#     'display_name': '测试库',
#     'description': '本地SQLite测试数据库'
# }


# ==================== 知识库配置 ====================
KNOWLEDGE_BASE_CONFIGS = {
    'hologres': {
        'file_path': '/Users/cherry/Desktop/数据库表结构/holo.xlsx',
        'sheet_name': '知识库'
    },
}


def get_database_config(db_name: str) -> Optional[Dict]:
    """获取指定数据库的配置"""
    return DATABASE_CONFIGS.get(db_name)


def get_available_databases() -> list:
    """获取所有可用数据库的列表"""
    return [
        {
            'id': db_id,
            'name': config['display_name'],
            'description': config['description']
        }
        for db_id, config in DATABASE_CONFIGS.items()
    ]


def get_knowledge_base_config(db_name: str) -> Dict:
    """获取指定数据库的知识库配置（支持上传覆盖）"""
    default_config = KNOWLEDGE_BASE_CONFIGS.get(db_name)
    if not default_config:
        # 返回空配置
        return {'file_path': None, 'sheet_name': '知识库'}
    
    # 检查是否有上传覆盖
    override = get_knowledge_override(db_name)
    if override and override.get('file_path') and os.path.isfile(override['file_path']):
        return {
            'file_path': override['file_path'],
            'sheet_name': override.get('sheet_name', default_config.get('sheet_name', '知识库'))
        }
    
    return default_config