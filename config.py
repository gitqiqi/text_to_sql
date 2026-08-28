# config.py - 精简版
import json
import os
import re
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
UPLOAD_MATCH_CONFIG_PATH = _PROJECT_ROOT / "upload_match_configs.json"


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


def _load_upload_match_meta() -> Dict:
    """加载上传匹配配置"""
    if not UPLOAD_MATCH_CONFIG_PATH.is_file():
        return {}
    try:
        return json.loads(UPLOAD_MATCH_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_upload_match_meta(meta: Dict) -> None:
    """保存上传匹配配置"""
    UPLOAD_MATCH_CONFIG_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_upload_template_label(value: str) -> str:
    """模板显示名的归一化版本，用于去重判断。"""
    return re.sub(r'\s+', ' ', str(value or '').strip()).lower()


def _is_upload_template_enabled(config: Dict) -> bool:
    value = (config or {}).get('is_enabled', True)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {'0', 'false', 'f', 'no', 'n', 'off', 'disabled'}


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

# 上传 Excel 后的业务匹配配置
# 默认建议只配置给后端，不在前端暴露字段名
UPLOAD_MATCH_CONFIGS = _load_upload_match_meta() or {
    # 'hologres': {
    #     'default': {
    #         'keyword_column': 'keyword',
    #         'match_mode': 'exact',
    #         'return_fields': [],
    #     },
    #     'sales_data': {
    #         'keyword_column': 'keyword',
    #         'match_table': 'public.customer_dim',
    #         'match_field': 'customer_name',
    #         'return_fields': [],
    #     },
    # }
}


def get_upload_match_configs() -> Dict:
    """获取全部上传匹配配置"""
    return _load_upload_match_meta() or UPLOAD_MATCH_CONFIGS


def save_upload_match_configs(meta: Dict) -> None:
    """保存全部上传匹配配置"""
    global UPLOAD_MATCH_CONFIGS
    UPLOAD_MATCH_CONFIGS = meta or {}
    _write_upload_match_meta(UPLOAD_MATCH_CONFIGS)


def get_upload_match_config_for_db(db_name: str) -> Dict:
    """获取指定数据库的上传匹配配置"""
    return get_upload_match_configs().get(db_name) or {}


def set_upload_match_config_for_db(db_name: str, db_config: Dict) -> None:
    """保存指定数据库的上传匹配配置"""
    meta = get_upload_match_configs()
    meta[db_name] = db_config or {}
    save_upload_match_configs(meta)


def delete_upload_match_template(db_name: str, template_key: str) -> None:
    """删除指定数据库下的模板"""
    meta = get_upload_match_configs()
    db_config = meta.get(db_name) or {}
    if template_key in db_config:
        db_config.pop(template_key, None)
        meta[db_name] = db_config
        save_upload_match_configs(meta)


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


def get_upload_match_config(
    db_name: str,
    template_key: Optional[str] = None,
    source_table_name: Optional[str] = None,
) -> Dict:
    """获取上传匹配配置（支持数据库默认配置 + 模板覆盖 + 表级覆盖）"""
    db_config = UPLOAD_MATCH_CONFIGS.get(db_name) or {}
    merged: Dict = {}

    default_config = db_config.get('default')
    if isinstance(default_config, dict):
        merged.update(default_config)

    if source_table_name and source_table_name != template_key:
        source_config = db_config.get(source_table_name)
        if isinstance(source_config, dict):
            merged.update(source_config)

    if template_key:
        template_config = db_config.get(template_key)
        if isinstance(template_config, dict):
            merged.update(template_config)

    return merged


def get_upload_match_templates(
    db_name: str,
    db_config: Optional[Dict] = None,
    enabled_only: bool = True,
) -> List[Dict]:
    """获取该数据库下可供前端展示的模板列表"""
    db_config = db_config if isinstance(db_config, dict) else (get_upload_match_configs().get(db_name) or {})
    template_items: List[tuple[str, Dict, str, str]] = []
    label_counts: Dict[str, int] = {}

    for template_key, template_config in db_config.items():
        if template_key == 'default' or not isinstance(template_config, dict):
            continue
        is_enabled = _is_upload_template_enabled(template_config)
        if enabled_only and not is_enabled:
            continue
        label = (template_config.get('label') or template_config.get('name') or template_key or '').strip()
        normalized = normalize_upload_template_label(label)
        if normalized:
            label_counts[normalized] = label_counts.get(normalized, 0) + 1
        template_items.append((template_key, template_config, label, normalized))

    templates: List[Dict] = []
    for template_key, template_config, label, normalized in template_items:
        return_fields = template_config.get('return_fields') if isinstance(template_config.get('return_fields'), list) else []
        enabled_count = 0
        for row in return_fields:
            if isinstance(row, dict) and row.get('enabled', True) is not False:
                enabled_count += 1
        duplicate_label = bool(normalized and label_counts.get(normalized, 0) > 1)
        templates.append({
            'template_key': template_key,
            'label': label or template_key,
            'label_display': f'{label}（{template_key}）' if duplicate_label and label else (label or template_key),
            'description': template_config.get('description') or '',
            'match_table': template_config.get('match_table') or '',
            'match_field': template_config.get('match_field') or '',
            'match_field_display': template_config.get('match_field_display') or template_config.get('match_field') or '',
            'match_mode': template_config.get('match_mode') or 'exact',
            'sql_text': template_config.get('sql_text') or '',
            'created_by': template_config.get('created_by') or '',
            'updated_by': template_config.get('updated_by') or '',
            'created_at': template_config.get('created_at') or '',
            'updated_at': template_config.get('updated_at') or '',
            'return_count': enabled_count,
            'is_enabled': is_enabled,
        })
    return templates
