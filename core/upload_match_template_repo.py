# core/upload_match_template_repo.py - 上传匹配模板的 knowledge 库持久化
import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import text

from .db_manager import DatabasePoolManager


_READY_DBS: set[str] = set()


def _json_dumps(value) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _json_loads_object(value: Any) -> Dict:
    if isinstance(value, dict):
        return dict(value)
    text_value = str(value or '').strip()
    if not text_value:
        return {}
    try:
        parsed = json.loads(text_value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_loads_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    text_value = str(value or '').strip()
    if not text_value:
        return []
    try:
        parsed = json.loads(text_value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text_value = str(value).strip().lower()
    if text_value in ('1', 'true', 't', 'yes', 'y', 'on', 'enabled'):
        return True
    if text_value in ('0', 'false', 'f', 'no', 'n', 'off', 'disabled'):
        return False
    return default


def _format_timestamp(value: Any) -> str:
    if not value:
        return ''
    if isinstance(value, datetime):
        return value.isoformat(sep=' ', timespec='seconds')
    return str(value)


def _parse_timestamp(value) -> Optional[datetime]:
    text_value = str(value or '').strip()
    if not text_value:
        return None
    try:
        return datetime.fromisoformat(text_value.replace('Z', '+00:00')).replace(tzinfo=None)
    except Exception:
        return None


def _display_user(user: Optional[Dict]) -> str:
    user = user or {}
    for key in ('user_name', 'mobile', 'we_user_id', 'employee_id', 'admin_id'):
        value = user.get(key)
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value:
            return text_value
    return 'system'


def ensure_upload_match_template_table(db_name: str) -> None:
    if db_name in _READY_DBS:
        return

    engine = DatabasePoolManager.get_engine(db_name)
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS knowledge.upload_match_template (
            db_name VARCHAR(50) NOT NULL,
            template_key VARCHAR(200) NOT NULL,
            label TEXT,
            description TEXT,
            keyword_column TEXT,
            match_table TEXT,
            match_field TEXT,
            match_field_display TEXT,
            match_mode TEXT,
            target_filter TEXT,
            sql_text TEXT,
            return_fields_json TEXT,
            config_json TEXT,
            status TEXT DEFAULT 'active',
            is_enabled BOOLEAN DEFAULT TRUE,
            created_by TEXT,
            updated_by TEXT,
            created_by_admin_id BIGINT,
            created_by_user_name TEXT,
            updated_by_admin_id BIGINT,
            updated_by_user_name TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (db_name, template_key)
        )
        """,
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS return_fields_json TEXT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS config_json TEXT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active'",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS is_enabled BOOLEAN DEFAULT TRUE",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_by TEXT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_by TEXT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_by_admin_id BIGINT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_by_user_name TEXT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_by_admin_id BIGINT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_by_user_name TEXT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()",
    ]
    dml_statements = [
        "UPDATE knowledge.upload_match_template SET status = 'active' WHERE status IS NULL",
        "UPDATE knowledge.upload_match_template SET is_enabled = TRUE WHERE is_enabled IS NULL",
    ]

    with engine.begin() as conn:
        for statement in ddl_statements:
            conn.execute(text(statement))

    with engine.begin() as conn:
        for statement in dml_statements:
            conn.execute(text(statement))
    _READY_DBS.add(db_name)


def _row_to_template_config(mapping: Dict[str, Any]) -> Dict:
    template_key = str(mapping.get('template_key') or '').strip()
    config = _json_loads_object(mapping.get('config_json'))

    for field in (
        'label',
        'description',
        'keyword_column',
        'match_table',
        'match_field',
        'match_field_display',
        'match_mode',
        'target_filter',
        'sql_text',
        'status',
    ):
        value = mapping.get(field)
        if value is not None:
            config[field] = str(value)
    config['is_enabled'] = _coerce_bool(
        mapping.get('is_enabled'),
        default=_coerce_bool(config.get('is_enabled'), True),
    )

    return_fields = _json_loads_list(mapping.get('return_fields_json'))
    if return_fields or 'return_fields' not in config:
        config['return_fields'] = return_fields

    config['label'] = str(config.get('label') or config.get('name') or template_key).strip()
    config['name'] = str(config.get('name') or config['label']).strip()

    for field in (
        'created_by',
        'updated_by',
        'created_by_admin_id',
        'created_by_user_name',
        'updated_by_admin_id',
        'updated_by_user_name',
        'created_at',
        'updated_at',
    ):
        value = mapping.get(field)
        if value not in (None, ''):
            config[field] = _format_timestamp(value)

    return config


def load_upload_match_templates_config(db_name: str) -> Dict[str, Dict]:
    """从 knowledge.upload_match_template 读取该库全部业务模板配置。"""
    ensure_upload_match_template_table(db_name)
    engine = DatabasePoolManager.get_engine(db_name)
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT
                    db_name,
                    template_key,
                    label,
                    description,
                    keyword_column,
                    match_table,
                    match_field,
                    match_field_display,
                    match_mode,
                    target_filter,
                    sql_text,
                    return_fields_json,
                    config_json,
                    status,
                    is_enabled,
                    created_by,
                    updated_by,
                    created_by_admin_id,
                    created_by_user_name,
                    updated_by_admin_id,
                    updated_by_user_name,
                    created_at,
                    updated_at
                FROM knowledge.upload_match_template
                WHERE db_name = :db_name
                  AND COALESCE(status, 'active') <> 'delete'
                ORDER BY
                    CASE WHEN template_key = 'default' THEN 0 ELSE 1 END,
                    updated_at DESC NULLS LAST,
                    template_key
            """),
            {'db_name': db_name},
        ).fetchall()

    configs: Dict[str, Dict] = {}
    for row in rows:
        mapping = dict(row._mapping)
        template_key = str(mapping.get('template_key') or '').strip()
        if not template_key:
            continue
        configs[template_key] = _row_to_template_config(mapping)
    return configs


def get_upload_match_config_from_db(
    db_name: str,
    template_key: Optional[str] = None,
    source_table_name: Optional[str] = None,
) -> Dict:
    """从数据库读取并合并默认配置、表级配置、模板配置。"""
    db_config = load_upload_match_templates_config(db_name)
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


def seed_upload_match_templates_from_config(
    db_name: str,
    db_config: Dict,
    current_user: Optional[Dict] = None,
) -> int:
    """把历史 JSON 配置导入数据库；已有模板会被 upsert。"""
    count = 0
    for template_key, config in (db_config or {}).items():
        template_key = str(template_key or '').strip()
        if not template_key or not isinstance(config, dict):
            continue
        upsert_upload_match_template(db_name, template_key, config, current_user=current_user)
        count += 1
    return count


def upsert_upload_match_template(
    db_name: str,
    template_key: str,
    config: Dict,
    current_user: Optional[Dict] = None,
) -> None:
    config = dict(config or {})
    actor = str(config.get('updated_by') or config.get('created_by') or _display_user(current_user)).strip() or 'system'
    user_name = (current_user or {}).get('user_name') or actor
    params = {
        'db_name': db_name,
        'template_key': template_key,
        'label': config.get('label') or config.get('name') or template_key,
        'description': config.get('description') or '',
        'keyword_column': config.get('keyword_column') or '',
        'match_table': config.get('match_table') or '',
        'match_field': config.get('match_field') or '',
        'match_field_display': config.get('match_field_display') or config.get('match_field') or '',
        'match_mode': config.get('match_mode') or 'exact',
        'target_filter': config.get('target_filter') or '',
        'sql_text': config.get('sql_text') or '',
        'return_fields_json': _json_dumps(config.get('return_fields') or []),
        'config_json': _json_dumps(config),
        'status': 'active',
        'is_enabled': _coerce_bool(config.get('is_enabled'), True),
        'created_by': str(config.get('created_by') or actor).strip() or 'system',
        'updated_by': actor,
        'created_by_admin_id': (current_user or {}).get('admin_id'),
        'created_by_user_name': user_name,
        'updated_by_admin_id': (current_user or {}).get('admin_id'),
        'updated_by_user_name': user_name,
        'created_at': _parse_timestamp(config.get('created_at')),
        'updated_at': _parse_timestamp(config.get('updated_at')),
    }

    ensure_upload_match_template_table(db_name)
    engine = DatabasePoolManager.get_engine(db_name)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO knowledge.upload_match_template (
                    db_name, template_key, label, description, keyword_column,
                    match_table, match_field, match_field_display, match_mode,
                    target_filter, sql_text, return_fields_json, config_json,
                    status, is_enabled, created_by, updated_by,
                    created_by_admin_id, created_by_user_name,
                    updated_by_admin_id, updated_by_user_name,
                    created_at, updated_at
                ) VALUES (
                    :db_name, :template_key, :label, :description, :keyword_column,
                    :match_table, :match_field, :match_field_display, :match_mode,
                    :target_filter, :sql_text, :return_fields_json, :config_json,
                    :status, :is_enabled, :created_by, :updated_by,
                    :created_by_admin_id, :created_by_user_name,
                    :updated_by_admin_id, :updated_by_user_name,
                    COALESCE(:created_at, NOW()), COALESCE(:updated_at, NOW())
                )
                ON CONFLICT (db_name, template_key) DO UPDATE SET
                    label = EXCLUDED.label,
                    description = EXCLUDED.description,
                    keyword_column = EXCLUDED.keyword_column,
                    match_table = EXCLUDED.match_table,
                    match_field = EXCLUDED.match_field,
                    match_field_display = EXCLUDED.match_field_display,
                    match_mode = EXCLUDED.match_mode,
                    target_filter = EXCLUDED.target_filter,
                    sql_text = EXCLUDED.sql_text,
                    return_fields_json = EXCLUDED.return_fields_json,
                    config_json = EXCLUDED.config_json,
                    status = EXCLUDED.status,
                    is_enabled = EXCLUDED.is_enabled,
                    created_by = COALESCE(knowledge.upload_match_template.created_by, EXCLUDED.created_by),
                    updated_by = EXCLUDED.updated_by,
                    created_by_admin_id = COALESCE(knowledge.upload_match_template.created_by_admin_id, EXCLUDED.created_by_admin_id),
                    created_by_user_name = COALESCE(knowledge.upload_match_template.created_by_user_name, EXCLUDED.created_by_user_name),
                    updated_by_admin_id = EXCLUDED.updated_by_admin_id,
                    updated_by_user_name = EXCLUDED.updated_by_user_name,
                    updated_at = COALESCE(EXCLUDED.updated_at, NOW())
            """),
            params,
        )


def update_upload_match_template_enabled(
    db_name: str,
    template_key: str,
    is_enabled: Any,
    current_user: Optional[Dict] = None,
) -> None:
    actor = _display_user(current_user)
    user_name = (current_user or {}).get('user_name') or actor
    ensure_upload_match_template_table(db_name)
    engine = DatabasePoolManager.get_engine(db_name)
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE knowledge.upload_match_template
                SET is_enabled = :is_enabled,
                    updated_by = :updated_by,
                    updated_by_admin_id = :updated_by_admin_id,
                    updated_by_user_name = :updated_by_user_name,
                    updated_at = NOW()
                WHERE db_name = :db_name
                  AND template_key = :template_key
                  AND COALESCE(status, 'active') <> 'delete'
            """),
            {
                'db_name': db_name,
                'template_key': template_key,
                'is_enabled': _coerce_bool(is_enabled, True),
                'updated_by': actor,
                'updated_by_admin_id': (current_user or {}).get('admin_id'),
                'updated_by_user_name': user_name,
            },
        )


def delete_upload_match_template_record(
    db_name: str,
    template_key: str,
    current_user: Optional[Dict] = None,
) -> None:
    actor = _display_user(current_user)
    user_name = (current_user or {}).get('user_name') or actor
    ensure_upload_match_template_table(db_name)
    engine = DatabasePoolManager.get_engine(db_name)
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE knowledge.upload_match_template
                SET status = 'delete',
                    updated_by = :updated_by,
                    updated_by_admin_id = :updated_by_admin_id,
                    updated_by_user_name = :updated_by_user_name,
                    updated_at = NOW()
                WHERE db_name = :db_name AND template_key = :template_key
            """),
            {
                'db_name': db_name,
                'template_key': template_key,
                'updated_by': actor,
                'updated_by_admin_id': (current_user or {}).get('admin_id'),
                'updated_by_user_name': user_name,
            },
        )


def load_upload_match_template_audit_map(db_name: str) -> Dict[str, Dict]:
    ensure_upload_match_template_table(db_name)
    engine = DatabasePoolManager.get_engine(db_name)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT template_key, created_by, updated_by,
                           created_by_admin_id, created_by_user_name,
                           updated_by_admin_id, updated_by_user_name,
                           created_at, updated_at
                    FROM knowledge.upload_match_template
                    WHERE db_name = :db_name
                      AND COALESCE(status, 'active') <> 'delete'
                """),
                {'db_name': db_name},
            ).fetchall()
    except Exception:
        return {}

    audit_map: Dict[str, Dict] = {}
    for row in rows:
        mapping = row._mapping
        template_key = str(mapping.get('template_key') or '').strip()
        if not template_key:
            continue
        audit_map[template_key] = {
            'created_by': mapping.get('created_by') or mapping.get('created_by_user_name') or '',
            'updated_by': mapping.get('updated_by') or mapping.get('updated_by_user_name') or '',
            'created_by_admin_id': mapping.get('created_by_admin_id'),
            'created_by_user_name': mapping.get('created_by_user_name') or '',
            'updated_by_admin_id': mapping.get('updated_by_admin_id'),
            'updated_by_user_name': mapping.get('updated_by_user_name') or '',
            'created_at': str(mapping.get('created_at')) if mapping.get('created_at') else '',
            'updated_at': str(mapping.get('updated_at')) if mapping.get('updated_at') else '',
        }
    return audit_map
