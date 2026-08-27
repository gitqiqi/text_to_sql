# core/upload_match_template_repo.py - 上传匹配模板的 knowledge 库镜像
import json
from datetime import datetime
from typing import Dict, Optional

from sqlalchemy import text

from .db_manager import DatabasePoolManager


_READY_DBS: set[str] = set()


def _json_dumps(value) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


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
    statements = [
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
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_by TEXT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_by TEXT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_by_admin_id BIGINT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_by_user_name TEXT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_by_admin_id BIGINT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_by_user_name TEXT",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()",
        "ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()",
    ]

    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
    _READY_DBS.add(db_name)


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
                    created_by, updated_by,
                    created_by_admin_id, created_by_user_name,
                    updated_by_admin_id, updated_by_user_name,
                    created_at, updated_at
                ) VALUES (
                    :db_name, :template_key, :label, :description, :keyword_column,
                    :match_table, :match_field, :match_field_display, :match_mode,
                    :target_filter, :sql_text, :return_fields_json, :config_json,
                    :created_by, :updated_by,
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


def delete_upload_match_template_record(db_name: str, template_key: str) -> None:
    ensure_upload_match_template_table(db_name)
    engine = DatabasePoolManager.get_engine(db_name)
    with engine.begin() as conn:
        conn.execute(
            text("""
                DELETE FROM knowledge.upload_match_template
                WHERE db_name = :db_name AND template_key = :template_key
            """),
            {'db_name': db_name, 'template_key': template_key},
        )


def load_upload_match_template_audit_map(db_name: str) -> Dict[str, Dict]:
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
