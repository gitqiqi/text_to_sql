# blueprints/main.py
import os
import csv
import json
import math
import re
import time
import uuid
from datetime import date, datetime, timedelta
from io import StringIO
from urllib.parse import quote
from flask import render_template, request, jsonify, redirect, url_for
from flask import Response, stream_with_context
import pandas as pd
from sqlalchemy import bindparam, text
from werkzeug.utils import secure_filename
from . import main_bp
from config import get_available_databases
from core import (
    DatabaseManager, KnowledgeBase, SQLKnowledgeRepo, TextToSQLConverter,
    clean_sql, validate_sql_safety, monitor_function, _nl_query_limiter,
    insert_query_log, TableSchemaSearcher,
    authenticate_user, get_current_user, login_user, logout_user,
    public_user_payload, user_can_view_all,
    sync_app_users_from_source,
    sync_book_from_source,
    get_last_book_sync_result,
    is_book_sync_monitor_running,
)
from core.embedding_client import iter_embedding_models
from core.cancellation import registry as cancel_registry, CancelledError
from core.db_manager import DatabasePoolManager
from core.upload_match_template_repo import (
    delete_upload_match_template_record,
    get_upload_match_config_from_db,
    get_upload_match_templates,
    is_current_upload_template_key,
    load_upload_match_templates_config,
    normalize_upload_template_label,
    update_upload_match_template_enabled,
    upsert_upload_match_template,
)

INSERT_RESPONSE_ROW_LIMIT = 200
INSERT_DUPLICATE_MAX_PAGE_SIZE = 1000
INSERT_SESSION_TTL_SECONDS = 2 * 60 * 60
_insert_excel_sessions: dict[str, dict] = {}


def _json_safe_cell(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(value, date):
        return value.isoformat()
    item_method = getattr(value, 'item', None)
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass
    return value


def safe_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame.to_dict(orient='records') 的安全版本，将 NaN/NaT/Inf 替换为 None"""
    df = df.astype(object).where(df.notna(), None)
    for col in df.columns:
        for i, val in enumerate(df[col]):
            df.at[i, col] = _json_safe_cell(val)
    return df.to_dict(orient='records')


def _clean_name(value: str) -> str:
    return re.sub(r'[\s_]+', '', str(value or '').strip()).lower()


def _quote_ident(value: str) -> str:
    text_value = str(value).replace('"', '""')
    return f'"{text_value}"'


def _split_field_list(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r'[,\n，；;]+', raw)
    return [p.strip() for p in parts if p and p.strip()]


def _split_return_field_alias(raw: str) -> tuple[str, str]:
    text = str(raw or '').strip()
    if not text:
        return '', ''

    as_match = re.match(r'^(.*?)\s+[Aa][Ss]\s+(.+)$', text)
    if as_match:
        return as_match.group(1).strip(), as_match.group(2).strip()

    parts = text.rsplit(None, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    return text, text


def _normalize_return_field_spec(item) -> dict | None:
    if item is None:
        return None

    if isinstance(item, dict):
        source = str(
            item.get('db_field')
            or item.get('source')
            or item.get('expr')
            or item.get('field')
            or item.get('column')
            or item.get('name')
            or ''
        ).strip()
        label = str(
            item.get('business_field')
            or item.get('label')
            or item.get('alias')
            or item.get('display_name')
            or ''
        ).strip()
        enabled = item.get('enabled', True)
    else:
        raw = str(item).strip()
        if not raw:
            return None
        source, label = _split_return_field_alias(raw)
        enabled = True

    if not source and label:
        source = label
    if not label:
        label = source

    source = str(source or '').strip()
    label = str(label or '').strip()
    if not source and not label:
        return None

    return {
        'db_field': source,
        'business_field': label,
        'enabled': bool(enabled),
    }


def _normalize_return_field_specs(return_fields) -> list[dict]:
    if not return_fields:
        return []

    if isinstance(return_fields, str):
        return_fields = _split_field_list(return_fields)

    if isinstance(return_fields, dict):
        return_fields = [return_fields]

    if not isinstance(return_fields, (list, tuple, set)):
        return []

    specs = []
    for item in return_fields:
        spec = _normalize_return_field_spec(item)
        if spec:
            specs.append(spec)
    return specs


def _is_simple_identifier(value: str) -> bool:
    return bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', str(value or '').strip()))


def _rewrite_target_column_qualifiers(expression: str, target_columns: list[str] | None = None) -> str:
    """将目标字段统一限定到匹配查询使用的 t 别名。"""
    expr = str(expression or '').strip()
    column_map = {
        str(column).lower(): str(column)
        for column in (target_columns or [])
        if str(column).strip()
    }
    if not expr or not column_map:
        return expr

    pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\.("?[A-Za-z_][A-Za-z0-9_]*"?)')

    def replace_qualified(match: re.Match) -> str:
        column_token = match.group(2)
        column_name = column_token.strip('"').lower()
        if column_name not in column_map:
            return match.group(0)
        return f't.{column_token}'

    def transform_sql_segment(segment: str) -> str:
        segment = pattern.sub(replace_qualified, segment)

        quoted_pattern = re.compile(r'"([^"]+)"')

        def replace_quoted(match: re.Match) -> str:
            column_name = match.group(1).replace('""', '"')
            canonical = column_map.get(column_name.lower())
            if not canonical:
                return match.group(0)
            previous = segment[:match.start()].rstrip()[-1:]
            following = segment[match.end():].lstrip()[:1]
            if previous == '.' or following == '.':
                return match.group(0)
            return f't.{_quote_ident(canonical)}'

        segment = quoted_pattern.sub(replace_quoted, segment)
        bare_pattern = re.compile(r'(?<![\w."])([A-Za-z_][A-Za-z0-9_]*)(?![\w"])')

        def replace_bare(match: re.Match) -> str:
            canonical = column_map.get(match.group(1).lower())
            if not canonical:
                return match.group(0)
            previous = segment[:match.start()].rstrip()[-1:]
            following = segment[match.end():].lstrip()[:1]
            if previous == '.' or following == '.':
                return match.group(0)
            return f't.{_quote_ident(canonical)}'

        return bare_pattern.sub(replace_bare, segment)

    output = []
    segment_start = 0
    index = 0
    while index < len(expr):
        if expr[index] != "'":
            index += 1
            continue
        output.append(transform_sql_segment(expr[segment_start:index]))
        literal_start = index
        index += 1
        while index < len(expr):
            if expr[index] != "'":
                index += 1
                continue
            if index + 1 < len(expr) and expr[index + 1] == "'":
                index += 2
                continue
            index += 1
            break
        output.append(expr[literal_start:index])
        segment_start = index
    output.append(transform_sql_segment(expr[segment_start:]))
    return ''.join(output)


def _validate_target_filter(raw_filter: str) -> str:
    """校验模板中的目标表筛选条件，只允许单个 WHERE 谓词。"""
    expression = str(raw_filter or '').strip()
    if not expression:
        return ''
    expression = re.sub(r'^\s*where\b', '', expression, count=1, flags=re.IGNORECASE).strip()
    if not expression:
        return ''
    if len(expression) > 20000:
        raise ValueError('目标表筛选条件过长')
    if ';' in expression or '--' in expression or '/*' in expression or '*/' in expression:
        raise ValueError('目标表筛选条件不能包含分号或 SQL 注释')

    scrubbed = []
    quote_char = ''
    depth = 0
    index = 0
    while index < len(expression):
        char = expression[index]
        if quote_char:
            scrubbed.append(' ')
            if char == quote_char:
                if index + 1 < len(expression) and expression[index + 1] == quote_char:
                    scrubbed.append(' ')
                    index += 2
                    continue
                quote_char = ''
            index += 1
            continue
        if char in {"'", '"'}:
            quote_char = char
            scrubbed.append(' ')
        else:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth < 0:
                    raise ValueError('目标表筛选条件括号不匹配')
            scrubbed.append(char)
        index += 1

    if quote_char or depth != 0:
        raise ValueError('目标表筛选条件中的引号或括号不完整')

    unsafe_sql = ''.join(scrubbed).lower()
    forbidden = (
        'select', 'insert', 'update', 'delete', 'drop', 'alter', 'create',
        'truncate', 'grant', 'revoke', 'copy', 'call', 'execute', 'merge',
        'vacuum', 'analyze', 'union', 'order', 'group', 'having', 'limit', 'offset',
    )
    if any(re.search(rf'\b{keyword}\b', unsafe_sql) for keyword in forbidden):
        raise ValueError('目标表筛选条件只能填写 WHERE 后面的过滤表达式')
    return expression


def _render_return_field_expression(source_expr: str, target_columns: list[str] | None = None) -> str:
    expr = str(source_expr or '').strip()
    if not expr:
        return ''
    if _is_simple_identifier(expr):
        return f't.{_quote_ident(expr)}'
    return _rewrite_target_column_qualifiers(expr, target_columns)


def _normalize_template_select_sql(sql_text: str) -> str:
    sql = clean_sql(sql_text)
    if not sql:
        return ''

    sql = sql.strip()
    while sql.endswith(';'):
        sql = sql[:-1].strip()
    if ';' in sql:
        raise ValueError('模板 SQL 只支持单条 SELECT/WITH 查询')
    if not re.match(r'^(select|with)\b', sql, flags=re.IGNORECASE):
        raise ValueError('模板 SQL 只支持 SELECT/WITH 查询')
    validate_sql_safety(sql)
    return sql


def _render_template_query_field_expression(spec: dict) -> str:
    field_name = str(spec.get('business_field') or spec.get('db_field') or '').strip()
    if not field_name:
        return ''
    return f't.{_quote_ident(field_name)}'


def _format_match_error(error: Exception) -> str:
    message = str(getattr(error, 'orig', None) or error).strip()
    if '[SQL:' in message:
        message = message.split('[SQL:', 1)[0].strip()
    return message or '匹配失败'


def _build_return_field_lookup(return_field_specs: list[dict]) -> dict[str, dict]:
    lookup = {}
    for spec in return_field_specs or []:
        if not isinstance(spec, dict):
            continue
        db_field = str(spec.get('db_field') or '').strip()
        business_field = str(spec.get('business_field') or '').strip()
        if db_field:
            lookup[_clean_name(db_field)] = spec
        if business_field:
            lookup[_clean_name(business_field)] = spec
    return lookup


def _resolve_target_field_spec(
    field_name: str,
    return_field_specs: list[dict],
    target_columns: list[str] | None = None,
) -> dict | None:
    raw = str(field_name or '').strip()
    if not raw:
        return None

    lookup = _build_return_field_lookup(return_field_specs)
    spec = lookup.get(_clean_name(raw))
    if spec:
        return spec

    if target_columns and raw in target_columns:
        return {
            'db_field': raw,
            'business_field': raw,
            'enabled': True,
        }

    return None


def _normalize_match_field_mappings(
    raw_mappings,
    keyword_column: str = '',
    match_field: str = '',
) -> list[dict]:
    mappings: list[dict] = []

    def add_mapping(source_value='', target_value='') -> None:
        source = str(source_value or '').strip().replace(' ', '_')
        target = str(target_value or '').strip()
        if not source and not target:
            return
        mappings.append({
            'source_field': source,
            'target_field': target,
        })

    items = raw_mappings
    if isinstance(raw_mappings, str):
        text_value = raw_mappings.strip()
        if not text_value:
            items = []
        else:
            try:
                parsed = json.loads(text_value)
                items = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                items = [text_value]
    elif isinstance(raw_mappings, dict):
        items = [raw_mappings]
    elif not isinstance(raw_mappings, list):
        items = []

    for item in items:
        if isinstance(item, dict):
            add_mapping(
                item.get('source_field') or item.get('keyword_column') or item.get('source') or item.get('left'),
                item.get('target_field') or item.get('match_field') or item.get('target') or item.get('right'),
            )
            continue

        text_value = str(item or '').strip()
        if not text_value:
            continue
        if '->' in text_value:
            left, right = text_value.split('->', 1)
            add_mapping(left, right)
        else:
            add_mapping(text_value, '')

    if mappings:
        if keyword_column and not mappings[0].get('source_field'):
            mappings[0]['source_field'] = str(keyword_column or '').strip().replace(' ', '_')
        if match_field and not mappings[0].get('target_field'):
            mappings[0]['target_field'] = str(match_field or '').strip()
        return mappings

    if keyword_column or match_field:
        add_mapping(keyword_column, match_field)
    return mappings


def _build_match_target_select_specs(
    match_field_specs,
    return_field_specs: list[dict],
    target_columns: list[str] | None = None,
    use_template_query: bool = False,
) -> tuple[list[dict], list[dict], dict[int, str]]:
    if isinstance(match_field_specs, dict):
        raw_match_specs = [match_field_specs]
    elif isinstance(match_field_specs, list):
        raw_match_specs = match_field_specs
    else:
        raw_match_specs = []

    match_specs = [dict(spec) for spec in raw_match_specs if isinstance(spec, dict)]
    if not match_specs:
        return [], [], {}

    match_alias_keys: dict[int, str] = {}
    select_specs: list[dict] = []
    seen_aliases: set[str] = set()

    def append_spec(spec: dict, match_index: int | None = None) -> None:
        if not isinstance(spec, dict) or not spec.get('enabled', True):
            return
        db_field = str(spec.get('db_field') or '').strip()
        business_field = str(spec.get('business_field') or '').strip() or db_field
        if not db_field and not business_field:
            return
        alias = business_field or db_field
        alias_key = _clean_name(alias)
        if match_index is not None:
            match_alias_keys[match_index] = alias_key
        if alias_key in seen_aliases:
            return

        expr = (
            _render_template_query_field_expression({'db_field': db_field, 'business_field': business_field})
            if use_template_query
            else _render_return_field_expression(db_field, target_columns)
        )
        if not expr:
            expr = 'NULL'

        select_specs.append({
            'db_field': db_field,
            'business_field': business_field,
            'expression': expr,
            'alias_key': alias_key,
        })
        seen_aliases.add(alias_key)

    for index, spec in enumerate(match_specs):
        append_spec(spec, index)
    for spec in return_field_specs or []:
        append_spec(spec)

    return match_specs, select_specs, match_alias_keys


def _build_timestamped_table_name(base_name: str, timestamp: str, extra_suffix: str = '', max_length: int = 63) -> str:
    safe_base = ''.join(c for c in str(base_name or '').replace(' ', '_').replace('-', '_') if c.isalnum() or c == '_')
    if not safe_base:
        return ''

    tail = f'_{timestamp}{extra_suffix}'
    max_base_len = max_length - len(tail)
    if max_base_len < 1:
        return ''
    safe_base = safe_base[:max_base_len].rstrip('_')
    if not safe_base:
        return ''

    return f'{safe_base}{tail}'


_UPLOAD_TABLE_PATTERN = re.compile(r'^(?P<base>.+)_(?P<ts>\d{8}_\d{6})(?:_(?P<legacy_suffix>\d{6}))?$')


def _parse_upload_table_name(table_name: str) -> dict | None:
    raw = str(table_name or '').strip()
    if not raw:
        return None
    if raw.startswith('tmp.'):
        raw = raw.split('.', 1)[1]
    if raw.endswith('_matched'):
        return None

    match = _UPLOAD_TABLE_PATTERN.match(raw)
    if not match:
        return None

    timestamp = match.group('ts')
    legacy_suffix = match.group('legacy_suffix') or ''
    return {
        'table_name': raw,
        'base_name': match.group('base'),
        'timestamp': timestamp,
        'legacy_suffix': legacy_suffix,
        'sort_key': f"{timestamp}_{legacy_suffix or '000000'}",
    }


def _split_schema_table(table_input: str, default_schema: str = 'tmp') -> tuple[str, str]:
    raw = str(table_input or '').strip()
    if '.' in raw:
        schema, table_name = raw.split('.', 1)
        return schema.strip().strip('"'), table_name.strip().strip('"')
    return default_schema, raw.strip().strip('"')


def _get_live_table_meta(db_name: str, table_input: str, default_schema: str = 'tmp') -> dict | None:
    schema, table_name = _split_schema_table(table_input, default_schema=default_schema)
    if not schema or not table_name:
        return None

    engine = DatabasePoolManager.get_engine(db_name)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table_name
            ORDER BY ordinal_position
        """), {'schema': schema, 'table_name': table_name}).fetchall()

    columns = [str(row._mapping.get('column_name')) for row in rows if row._mapping.get('column_name')]
    if not columns:
        return None
    return {
        'schema': schema,
        'table_name': table_name,
        'columns': columns,
    }


def _get_cached_upload_tables(db_name: str, limit: int) -> list[dict]:
    tables = KnowledgeBase(db_name).get_table_list()
    history = []

    for table in tables:
        if str(table.get('schema') or '') != 'tmp':
            continue
        table_name = str(table.get('table_name') or '').strip()
        parsed = _parse_upload_table_name(table_name)
        if not parsed:
            continue
        history.append({
            'schema': 'tmp',
            'table_name': parsed['table_name'],
            'label': parsed['timestamp'],
            'base_name': parsed['base_name'],
            'timestamp': parsed['timestamp'],
            'legacy_suffix': parsed['legacy_suffix'],
            'column_count': 0,
            'columns': [],
            'sort_key': parsed['sort_key'],
        })

    history.sort(key=lambda item: (item['sort_key'], item['table_name']), reverse=True)
    return history[:limit]


def _get_live_upload_tables(db_name: str, limit: int) -> list[dict]:
    engine = DatabasePoolManager.get_engine(db_name)
    with engine.connect() as conn:
        table_rows = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'tmp'
            ORDER BY table_name DESC
        """)).fetchall()

    history = []
    for row in table_rows:
        mapping = row._mapping
        table_name = str(mapping.get('table_name') or '').strip()
        parsed = _parse_upload_table_name(table_name)
        if not parsed:
            continue
        history.append({
            'schema': 'tmp',
            'table_name': parsed['table_name'],
            'label': parsed['timestamp'],
            'base_name': parsed['base_name'],
            'timestamp': parsed['timestamp'],
            'legacy_suffix': parsed['legacy_suffix'],
            'column_count': 0,
            'columns': [],
            'sort_key': parsed['sort_key'],
        })

    history.sort(key=lambda item: (item['sort_key'], item['table_name']), reverse=True)
    return history[:limit]


def _get_live_database_schemas(db_name: str) -> list[str]:
    engine = DatabasePoolManager.get_engine(db_name)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT table_schema
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema', 'knowledge', 'hg_recyclebin')
              AND table_schema NOT LIKE 'pg_%'
            ORDER BY table_schema
        """)).fetchall()
    return [str(row._mapping.get('table_schema') or '').strip() for row in rows if row._mapping.get('table_schema')]


def _get_live_database_tables(
    db_name: str,
    schema_names: list[str] | None = None,
    include_columns: bool = True,
) -> list[dict]:
    """从数据库实时读取可访问的表和字段，补齐知识库尚未同步的表。"""
    schemas = [str(schema or '').strip() for schema in (schema_names or []) if str(schema or '').strip()]
    schema_clause = "AND table_schema IN :schema_names" if schemas else ""

    if not include_columns:
        query = text(f"""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema', 'knowledge', 'hg_recyclebin')
              AND table_schema NOT LIKE 'pg_%'
              AND table_name NOT LIKE '%_middle%'
              {schema_clause}
            ORDER BY table_schema, table_name
        """)
        params = {}
        if schemas:
            query = query.bindparams(bindparam('schema_names', expanding=True))
            params['schema_names'] = schemas

        engine = DatabasePoolManager.get_engine(db_name)
        with engine.connect() as conn:
            rows = conn.execute(query, params).fetchall()

        tables = []
        for row in rows:
            mapping = row._mapping
            schema = str(mapping.get('table_schema') or '').strip()
            table_name = str(mapping.get('table_name') or '').strip()
            if not schema or not table_name:
                continue
            tables.append({
                'schema': schema,
                'table_name': table_name,
                'columns': [],
                'column_count': 0,
                'columns_loaded': False,
            })
        return tables

    query = text(f"""
        SELECT table_schema, table_name, column_name, ordinal_position
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema', 'knowledge', 'hg_recyclebin')
          AND table_schema NOT LIKE 'pg_%'
          AND table_name NOT LIKE '%_middle%'
          {schema_clause}
        ORDER BY table_schema, table_name, ordinal_position
    """)
    params = {}
    if schemas:
        query = query.bindparams(bindparam('schema_names', expanding=True))
        params['schema_names'] = schemas

    engine = DatabasePoolManager.get_engine(db_name)
    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    grouped = {}
    for row in rows:
        mapping = row._mapping
        schema = str(mapping.get('table_schema') or '').strip()
        table_name = str(mapping.get('table_name') or '').strip()
        if not schema or not table_name:
            continue
        entry = grouped.setdefault((schema, table_name), {
            'schema': schema,
            'table_name': table_name,
            'columns': [],
            'column_count': 0,
            'columns_loaded': True,
        })
        column_name = mapping.get('column_name')
        if column_name:
            entry['columns'].append(str(column_name))

    tables = list(grouped.values())
    for table in tables:
        table['column_count'] = len(table['columns'])
    return tables


def _format_full_table_name(table_meta: dict) -> str:
    schema = table_meta.get('schema')
    table_name = table_meta.get('table_name')
    return f"{schema}.{table_name}" if schema else str(table_name)


def _quote_table_name(table_meta: dict) -> str:
    schema = table_meta.get('schema')
    table_name = table_meta.get('table_name')
    if schema:
        return f'{_quote_ident(schema)}.{_quote_ident(table_name)}'
    return _quote_ident(table_name)


def _resolve_table_meta(tables: list[dict], table_input: str) -> dict | None:
    if not table_input:
        return None

    raw = table_input.strip()
    if not raw:
        return None

    normalized = _clean_name(raw)
    if '.' in raw:
        schema_part, table_part = raw.split('.', 1)
        schema_norm = _clean_name(schema_part)
        table_norm = _clean_name(table_part)
        for table in tables:
            if _clean_name(table.get('schema')) == schema_norm and _clean_name(table.get('table_name')) == table_norm:
                return table
        return None

    same_name = [table for table in tables if _clean_name(table.get('table_name')) == normalized]
    if len(same_name) == 1:
        return same_name[0]
    if len(same_name) > 1:
        raise ValueError(f'目标表 {raw} 在多个 schema 中都存在，请使用 schema.table 形式')

    for table in tables:
        if _clean_name(_format_full_table_name(table)) == normalized:
            return table
    return None


def _normalize_insert_excel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """校验插入 Excel 的表头，并拒绝空列名或重复列名。"""
    normalized_columns = []
    seen_columns = set()
    for raw_column in df.columns:
        column = str(raw_column).strip()
        if not column or column.lower() == 'nan':
            raise ValueError('Excel 存在空列名，请先补充表头')
        normalized_key = column.lower()
        if normalized_key in seen_columns:
            raise ValueError(f'Excel 存在重复列名：{column}')
        seen_columns.add(normalized_key)
        normalized_columns.append(column)

    df = df.copy()
    df.columns = normalized_columns
    df = df.dropna(how='all').reset_index(drop=True)
    if df.empty:
        raise ValueError('Excel 没有可插入的数据行')
    return df


def _resolve_insert_target_table_meta(db_name: str, table_input: str) -> dict:
    """只解析 tmp schema 下的已有数据库表。"""
    raw_table_input = str(table_input or '').strip()
    if not raw_table_input:
        raise ValueError('请选择目标表')

    schema_name, table_name = _split_schema_table(raw_table_input, default_schema='tmp')
    if schema_name.lower() != 'tmp':
        raise ValueError('插入数据只允许操作 tmp schema 下的表')
    if not table_name:
        raise ValueError('请选择目标表')
    if '_middle' in table_name.lower():
        raise ValueError('插入数据不支持操作 _middle 中间表')

    resolved_meta = _get_live_table_meta(
        db_name,
        f'tmp.{table_name}',
        default_schema='tmp',
    )
    if not resolved_meta or not resolved_meta.get('columns'):
        raise ValueError(f'未找到 tmp 表：tmp.{table_name}')
    return resolved_meta


def _prepare_insert_excel_data(
    db_name: str,
    table_input: str,
    file,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """读取 Excel，要求 Excel 字段是目标表字段的严格同名子集。"""
    if not file or not file.filename:
        raise ValueError('请选择文件')
    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        raise ValueError('仅支持 .xlsx 或 .xls 格式')

    df = pd.read_excel(file)
    if df.empty or len(df.columns) == 0:
        raise ValueError('Excel 文件为空')
    df = _normalize_insert_excel_columns(df)

    target_meta = _resolve_insert_target_table_meta(db_name, table_input)
    target_columns = list(target_meta.get('columns') or [])
    unknown_columns = [
        column for column in df.columns
        if column not in target_columns
    ]
    if unknown_columns:
        raise ValueError(
            f'Excel 字段必须与 tmp 目标表字段同名，未知字段：{"、".join(unknown_columns)}'
        )

    column_mappings = [
        {
            'excel_column': column,
            'target_column': column,
        }
        for column in df.columns
    ]
    return target_meta, df, column_mappings


def _insert_staging_table_name() -> str:
    return f'__insert_stage_{uuid.uuid4().hex[:16]}'


def _insert_staging_table_sql(target_meta: dict, staging_table_name: str) -> str:
    return (
        f'{_quote_ident(target_meta.get("schema") or "tmp")}.'
        f'{_quote_ident(staging_table_name)}'
    )


def _insert_drop_table_suffix(engine) -> str:
    return '' if getattr(engine.dialect, 'name', '') == 'sqlite' else ' CASCADE'


def _create_insert_staging_table(
    db_name: str,
    target_meta: dict,
    df: pd.DataFrame,
) -> str:
    """把整份 Excel 落到 tmp staging 表，后续查重和插入都在库内完成。"""
    staging_table_name = _insert_staging_table_name()
    target_table_sql = _quote_table_name(target_meta)
    staging_table_sql = _insert_staging_table_sql(target_meta, staging_table_name)
    staging_columns = ', '.join(
        [
            'CAST(NULL AS BIGINT) AS "__upload_row_index"',
            'CAST(NULL AS BOOLEAN) AS "__insert_existing_match"',
            'CAST(NULL AS BOOLEAN) AS "__insert_file_duplicate"',
            *[
                f't.{_quote_ident(column)} AS {_quote_ident(column)}'
                for column in df.columns
            ],
        ]
    )
    create_sql = f"""
        CREATE TABLE {staging_table_sql} AS
        SELECT {staging_columns}
        FROM {target_table_sql} t
        WHERE 1 = 0
    """

    engine = DatabasePoolManager.get_engine(db_name)
    try:
        # Hologres 对 DDL/DML 的事务边界比较严格，建表和写入分开提交。
        with engine.connect() as conn:
            conn.execute(text(
                f'DROP TABLE IF EXISTS {staging_table_sql}{_insert_drop_table_suffix(engine)}'
            ))
            conn.execute(text(create_sql))
            conn.commit()

        staging_df = df.copy()
        staging_df.insert(0, '__upload_row_index', range(len(staging_df)))
        staging_df.to_sql(
            staging_table_name,
            engine,
            schema=target_meta.get('schema') or 'tmp',
            if_exists='append',
            index=False,
            method=None,
            chunksize=5000,
        )
    except Exception:
        _drop_insert_staging_table(db_name, target_meta, staging_table_name)
        raise

    return staging_table_name


def _drop_insert_staging_table(
    db_name: str,
    target_meta: dict,
    staging_table_name: str | None,
):
    if not staging_table_name:
        return
    try:
        engine = DatabasePoolManager.get_engine(db_name)
        staging_table_sql = _insert_staging_table_sql(target_meta, staging_table_name)
        with engine.connect() as conn:
            conn.execute(text(
                f'DROP TABLE IF EXISTS {staging_table_sql}{_insert_drop_table_suffix(engine)}'
            ))
            conn.commit()
    except Exception as error:
        print(f'清理插入 staging 表失败（忽略）: {error}')


def _get_insert_table_constraints(db_name: str, target_meta: dict) -> list[dict]:
    """读取 tmp 表的主键和唯一约束，优先返回主键。"""
    engine = DatabasePoolManager.get_engine(db_name)
    schema_name = str(target_meta.get('schema') or '').strip()
    table_name = str(target_meta.get('table_name') or '').strip()
    if not schema_name or not table_name:
        return []

    if getattr(engine.dialect, 'name', '') == 'sqlite':
        constraints = []
        with engine.connect() as conn:
            table_identifier = table_name.replace('"', '""')
            primary_rows = conn.exec_driver_sql(
                f'PRAGMA table_info("{table_identifier}")'
            ).fetchall()
        primary_columns = [
            str(row[1])
            for row in sorted(primary_rows, key=lambda item: int(item[5] or 0))
            if int(row[5] or 0) > 0
        ]
        if primary_columns:
            constraints.append({
                'name': f'{table_name}_pkey',
                'type': 'PRIMARY KEY',
                'columns': primary_columns,
            })
        return constraints

    query = text("""
        SELECT
            tc.constraint_name,
            tc.constraint_type,
            kcu.column_name,
            kcu.ordinal_position
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_catalog = kcu.constraint_catalog
         AND tc.constraint_schema = kcu.constraint_schema
         AND tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
         AND tc.table_name = kcu.table_name
        WHERE tc.table_schema = :schema_name
          AND tc.table_name = :table_name
          AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
        ORDER BY
            CASE WHEN tc.constraint_type = 'PRIMARY KEY' THEN 0 ELSE 1 END,
            tc.constraint_name,
            kcu.ordinal_position
    """)
    grouped = {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                query,
                {
                    'schema_name': schema_name,
                    'table_name': table_name,
                },
            ).fetchall()
    except Exception:
        return []

    for row in rows:
        mapping = row._mapping
        name = str(mapping.get('constraint_name') or '').strip()
        column = str(mapping.get('column_name') or '').strip()
        constraint_type = str(mapping.get('constraint_type') or '').strip()
        if not name or not column:
            continue
        item = grouped.setdefault(
            (name, constraint_type),
            {
                'name': name,
                'type': constraint_type,
                'columns': [],
            },
        )
        item['columns'].append(column)

    return list(grouped.values())


def _resolve_insert_duplicate_key(
    db_name: str,
    target_meta: dict,
    upload_columns: list[str],
) -> dict:
    constraints = _get_insert_table_constraints(db_name, target_meta)
    full_table_name = _format_full_table_name(target_meta)
    if not constraints:
        raise ValueError(
            f'目标表 {full_table_name} 没有主键或唯一约束，无法可靠判断重复数据；'
            '请先为 tmp 表设置主键或唯一键'
        )

    upload_column_set = set(upload_columns)
    usable = [
        item for item in constraints
        if set(item.get('columns') or []).issubset(upload_column_set)
    ]
    if not usable:
        key_text = '；'.join(
            f'{item.get("type", "唯一键")} {", ".join(item.get("columns") or [])}'
            for item in constraints
        )
        raise ValueError(
            f'Excel 必须包含目标表的主键或唯一键字段，当前约束为：{key_text}'
        )
    return usable[0]


def _insert_join_condition(left_alias: str, right_alias: str, key_columns: list[str]) -> str:
    return ' AND '.join(
        f'({left_alias}.{_quote_ident(column_name)} = {right_alias}.{_quote_ident(column_name)} '
        f'OR ({left_alias}.{_quote_ident(column_name)} IS NULL '
        f'AND {right_alias}.{_quote_ident(column_name)} IS NULL))'
        for column_name in key_columns
    )

def _find_insert_duplicates_with_staging(
    db_name: str,
    target_meta: dict,
    df: pd.DataFrame,
    key_columns: list[str],
    staging_table_name: str,
) -> dict:
    """在 staging 表内标记并查出历史重复和 Excel 内重复行号。"""
    empty = {
        'existing_row_indices': set(),
        'file_row_indices': set(),
    }
    if df.empty or not key_columns or not staging_table_name:
        return empty

    engine = DatabasePoolManager.get_engine(db_name)
    target_table_sql = _quote_table_name(target_meta)
    staging_table_sql = _insert_staging_table_sql(target_meta, staging_table_name)
    key_projection = ', '.join(
        _quote_ident(column_name)
        for column_name in key_columns
    )
    target_join = _insert_join_condition('s', 't', key_columns)
    file_join = _insert_join_condition('s', 'd', key_columns)
    file_count = '__insert_file_count'

    mark_existing_query = text(f"""
        UPDATE {staging_table_sql} AS s
        SET {_quote_ident('__insert_existing_match')} = TRUE
        FROM {target_table_sql} t
        WHERE {target_join}
    """)
    mark_file_duplicate_query = text(f"""
        UPDATE {staging_table_sql} AS s
        SET {_quote_ident('__insert_file_duplicate')} = TRUE
        FROM (
            SELECT {key_projection}, COUNT(*) AS {_quote_ident(file_count)}
            FROM {staging_table_sql}
            GROUP BY {key_projection}
            HAVING COUNT(*) > 1
        ) d
        WHERE {file_join}
    """)
    query = text(f"""
        SELECT
            s."__upload_row_index" AS upload_row,
            CASE
                WHEN COALESCE(s."__insert_existing_match", FALSE) THEN 1
                ELSE 0
            END AS existing_match,
            CASE
                WHEN COALESCE(s."__insert_file_duplicate", FALSE) THEN 1
                ELSE 0
            END AS file_match
        FROM {staging_table_sql} s
        WHERE COALESCE(s."__insert_existing_match", FALSE)
           OR COALESCE(s."__insert_file_duplicate", FALSE)
        ORDER BY s."__upload_row_index"
    """)

    with engine.connect() as conn:
        try:
            conn.execute(mark_existing_query)
            conn.commit()
            conn.execute(mark_file_duplicate_query)
            conn.commit()
            rows = conn.execute(query).fetchall()
        except Exception:
            conn.rollback()
            raise

    existing_indices = set()
    file_indices = set()
    for row in rows:
        mapping = row._mapping
        upload_row = mapping.get('upload_row')
        if upload_row is None:
            continue
        upload_row = int(upload_row)
        if int(mapping.get('existing_match') or 0):
            existing_indices.add(upload_row)
        if int(mapping.get('file_match') or 0):
            file_indices.add(upload_row)
    return {
        'existing_row_indices': existing_indices,
        'file_row_indices': file_indices,
    }


def _fetch_insert_existing_duplicate_rows(
    db_name: str,
    target_meta: dict,
    df: pd.DataFrame,
    key_columns: list[str],
    row_indices: list[int],
    staging_table_name: str,
) -> pd.DataFrame:
    target_columns = list(target_meta.get('columns') or [])
    failed_columns = ['__upload_row_index', 'Excel行号'] + target_columns + ['失败原因']
    if df.empty or not key_columns or not row_indices or not staging_table_name:
        return pd.DataFrame(columns=failed_columns)

    engine = DatabasePoolManager.get_engine(db_name)
    target_table_sql = _quote_table_name(target_meta)
    staging_table_sql = _insert_staging_table_sql(target_meta, staging_table_name)
    target_projection = ', '.join(
        f't.{_quote_ident(column)}'
        for column in target_columns
    ) or 't.*'
    join_condition = _insert_join_condition('t', 's', key_columns)
    query = text(f"""
        SELECT
            s."__upload_row_index" AS "__upload_row_index",
            s."__upload_row_index" + 2 AS "Excel行号",
            {target_projection}
        FROM {staging_table_sql} s
        JOIN {target_table_sql} t
          ON {join_condition}
        WHERE COALESCE(s."__insert_existing_match", FALSE)
          AND s."__upload_row_index" IN :upload_indices
        ORDER BY s."__upload_row_index"
    """).bindparams(bindparam('upload_indices', expanding=True))

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {'upload_indices': [int(index) for index in row_indices]},
        ).fetchall()

    records = []
    for row in rows:
        record = dict(row._mapping)
        record['失败原因'] = '目标表已存在相同主键或唯一键'
        records.append(record)
    if not records:
        return pd.DataFrame(columns=failed_columns)

    failed_df = pd.DataFrame(records)
    for column in failed_columns:
        if column not in failed_df.columns:
            failed_df[column] = None
    return failed_df[failed_columns]


def _insert_non_duplicate_rows_from_staging(
    db_name: str,
    target_meta: dict,
    df: pd.DataFrame,
    staging_table_name: str,
) -> int:
    """从 staging 表直接 INSERT 可插入行，避免把非重复数据再传回数据库。"""
    if df.empty or not staging_table_name:
        return 0

    engine = DatabasePoolManager.get_engine(db_name)
    target_table_sql = _quote_table_name(target_meta)
    staging_table_sql = _insert_staging_table_sql(target_meta, staging_table_name)
    insert_columns = list(df.columns)
    insert_projection = ', '.join(
        f's.{_quote_ident(column)}'
        for column in insert_columns
    )
    insert_column_names = ', '.join(
        _quote_ident(column)
        for column in insert_columns
    )
    query = text(f"""
        INSERT INTO {target_table_sql} ({insert_column_names})
        SELECT {insert_projection}
        FROM {staging_table_sql} s
        WHERE COALESCE(s."__insert_existing_match", FALSE) = FALSE
          AND COALESCE(s."__insert_file_duplicate", FALSE) = FALSE
    """)
    with engine.begin() as conn:
        result = conn.execute(query)
        rowcount = result.rowcount
    return int(rowcount) if rowcount and rowcount > 0 else 0


def _analyze_insert_duplicates(
    db_name: str,
    target_meta: dict,
    df: pd.DataFrame,
    staging_table_name: str,
) -> dict:
    if not staging_table_name:
        raise ValueError('插入查重必须使用 staging 会话，请重新上传 Excel')

    duplicate_key = _resolve_insert_duplicate_key(
        db_name,
        target_meta,
        list(df.columns),
    )
    key_columns = list(duplicate_key.get('columns') or [])
    staging_summary = _find_insert_duplicates_with_staging(
        db_name,
        target_meta,
        df,
        key_columns,
        staging_table_name,
    )
    existing_duplicate_indices = set(
        staging_summary.get('existing_row_indices') or []
    )
    file_duplicate_indices = set(
        staging_summary.get('file_row_indices') or []
    )
    duplicate_indices = existing_duplicate_indices | file_duplicate_indices
    return {
        'key_name': duplicate_key.get('name') or '',
        'key_type': duplicate_key.get('type') or '',
        'key_columns': key_columns,
        'existing_rows': len(existing_duplicate_indices),
        'file_rows': len(file_duplicate_indices),
        'total_rows': len(duplicate_indices),
        'existing_row_indices': sorted(existing_duplicate_indices),
        'file_row_indices': sorted(file_duplicate_indices),
        'row_indices': sorted(duplicate_indices),
        'staging_table_name': staging_table_name or '',
    }


def _public_insert_duplicate_summary(duplicate_summary: dict) -> dict:
    """只返回分页和更新状态需要的重复汇总。"""
    return {
        'key_name': duplicate_summary.get('key_name') or '',
        'key_type': duplicate_summary.get('key_type') or '',
        'key_columns': list(duplicate_summary.get('key_columns') or []),
        'existing_rows': int(duplicate_summary.get('existing_rows') or 0),
        'file_rows': int(duplicate_summary.get('file_rows') or 0),
        'total_rows': int(duplicate_summary.get('total_rows') or 0),
    }


def _parse_insert_duplicate_page_options(source) -> tuple[int, int]:
    try:
        page = int(source.get('duplicate_page') or source.get('page') or 1)
    except Exception:
        page = 1
    try:
        page_size = int(
            source.get('duplicate_page_size')
            or source.get('page_size')
            or INSERT_RESPONSE_ROW_LIMIT
        )
    except Exception:
        page_size = INSERT_RESPONSE_ROW_LIMIT

    page = max(1, page)
    page_size = max(1, min(page_size, INSERT_DUPLICATE_MAX_PAGE_SIZE))
    return page, page_size


def _build_insert_duplicate_page_meta(
    duplicate_summary: dict,
    page: int,
    page_size: int,
) -> dict:
    total = int(duplicate_summary.get('total_rows') or 0)
    total_pages = math.ceil(total / page_size) if total else 0
    normalized_page = min(page, total_pages) if total_pages else 1
    offset = (normalized_page - 1) * page_size if total else 0
    current_count = max(0, min(page_size, total - offset))
    return {
        'page': normalized_page,
        'page_size': page_size,
        'total': total,
        'total_pages': total_pages,
        'offset': offset,
        'current_count': current_count,
        'has_prev': total_pages > 0 and normalized_page > 1,
        'has_next': total_pages > 0 and normalized_page < total_pages,
    }


def _cleanup_insert_excel_sessions():
    now = time.time()
    expired = [
        session_id for session_id, item in _insert_excel_sessions.items()
        if float(item.get('expires_at') or 0) <= now
    ]
    for session_id in expired:
        item = _insert_excel_sessions.get(session_id) or {}
        _drop_insert_staging_table(
            item.get('db_name') or '',
            item.get('target_meta') or {},
            item.get('staging_table_name'),
        )
        _insert_excel_sessions.pop(session_id, None)


def _store_insert_excel_session(
    db_name: str,
    target_meta: dict,
    df: pd.DataFrame,
    column_mappings: list[dict],
    duplicate_summary: dict,
    staging_table_name: str,
) -> str:
    _cleanup_insert_excel_sessions()
    session_id = uuid.uuid4().hex
    now = time.time()
    _insert_excel_sessions[session_id] = {
        'db_name': db_name,
        'target_meta': dict(target_meta),
        'target_table': _format_full_table_name(target_meta),
        'df': df.copy(),
        'column_mappings': list(column_mappings or []),
        'duplicate_summary': duplicate_summary,
        'staging_table_name': staging_table_name,
        'created_at': now,
        'expires_at': now + INSERT_SESSION_TTL_SECONDS,
    }
    return session_id


def _delete_insert_excel_session(session_id: str):
    session_id = str(session_id or '').strip()
    if not session_id:
        return
    item = _insert_excel_sessions.pop(session_id, None)
    if not item:
        return
    _drop_insert_staging_table(
        item.get('db_name') or '',
        item.get('target_meta') or {},
        item.get('staging_table_name'),
    )


def _get_insert_excel_session(session_id: str, db_name: str, table_name: str = '') -> dict | None:
    session_id = str(session_id or '').strip()
    if not session_id:
        return None

    _cleanup_insert_excel_sessions()
    item = _insert_excel_sessions.get(session_id)
    if not item:
        raise ValueError('重复数据会话已过期，请重新上传 Excel 后再操作')
    if db_name and item.get('db_name') != db_name:
        raise ValueError('重复数据会话与当前数据库不一致，请重新上传 Excel')
    if table_name:
        schema_name, raw_table_name = _split_schema_table(table_name, default_schema='tmp')
        requested_table = f'{schema_name}.{raw_table_name}' if schema_name else raw_table_name
        if _clean_name(requested_table) != _clean_name(item.get('target_table')):
            raise ValueError('重复数据会话与目标表不一致，请重新上传 Excel')

    item['expires_at'] = time.time() + INSERT_SESSION_TTL_SECONDS
    return item


def _build_insert_duplicate_response_payload(
    db_name: str,
    target_meta: dict,
    df: pd.DataFrame,
    column_mappings: list[dict],
    duplicate_summary: dict,
    page: int,
    page_size: int,
    *,
    mode: str,
    insert_session_id: str,
) -> dict:
    page_meta = _build_insert_duplicate_page_meta(duplicate_summary, page, page_size)
    failed_df = _build_insert_failed_rows(
        db_name,
        target_meta,
        df,
        duplicate_summary,
        offset=page_meta['offset'],
        limit=page_meta['page_size'],
    )
    update_columns, update_rows = _build_insert_duplicate_update_rows(
        db_name,
        target_meta,
        df,
        duplicate_summary,
        offset=page_meta['offset'],
        limit=page_meta['page_size'],
    )
    duplicate_count = int(duplicate_summary.get('total_rows') or 0)
    return {
        'mode': mode,
        'table_name': _format_full_table_name(target_meta),
        'insert_session_id': insert_session_id,
        'duplicate_summary': _public_insert_duplicate_summary(duplicate_summary),
        'duplicate_page': page_meta['page'],
        'duplicate_page_size': page_meta['page_size'],
        'duplicate_total_pages': page_meta['total_pages'],
        'duplicate_page_offset': page_meta['offset'],
        'duplicate_page_count': page_meta['current_count'],
        'duplicate_has_prev': page_meta['has_prev'],
        'duplicate_has_next': page_meta['has_next'],
        'column_mappings': column_mappings,
        'failed_columns': list(failed_df.columns),
        'failed_rows': safe_records(failed_df),
        'failed_count': duplicate_count,
        'failed_preview_count': len(failed_df),
        'failed_rows_limited': duplicate_count > len(failed_df),
        'duplicate_update_columns': update_columns,
        'duplicate_update_rows': update_rows,
        'duplicate_update_total': duplicate_count,
        'insert_update_eligible_total': int(
            duplicate_summary.get('existing_rows') or 0
        ),
        'duplicate_update_preview_count': len(update_rows),
        'problem_rows': duplicate_count,
    }


def _build_insert_failed_rows(
    db_name: str,
    target_meta: dict,
    df: pd.DataFrame,
    duplicate_summary: dict,
    offset: int = 0,
    limit: int | None = None,
) -> pd.DataFrame:
    duplicate_indices = list(duplicate_summary.get('row_indices') or [])
    if not duplicate_indices:
        return df.iloc[0:0]

    existing_indices = set(duplicate_summary.get('existing_row_indices') or [])
    file_indices = set(duplicate_summary.get('file_row_indices') or [])
    key_columns = list(duplicate_summary.get('key_columns') or [])
    existing_row_indices = sorted(existing_indices)
    file_only_indices = sorted(file_indices - existing_indices)
    offset = max(0, int(offset or 0))
    if limit is not None:
        limit = max(0, int(limit))
        existing_count = len(existing_row_indices)
        existing_start = min(offset, existing_count)
        existing_end = min(existing_count, offset + limit)
        selected_existing_indices = existing_row_indices[existing_start:existing_end]
        remaining_limit = max(0, limit - len(selected_existing_indices))
        file_start = max(0, offset - existing_count)
        selected_file_only_indices = file_only_indices[file_start:file_start + remaining_limit]
    else:
        existing_count = len(existing_row_indices)
        selected_existing_indices = existing_row_indices[offset:]
        file_start = max(0, offset - existing_count)
        selected_file_only_indices = file_only_indices[file_start:]

    existing_failed_df = _fetch_insert_existing_duplicate_rows(
        db_name,
        target_meta,
        df,
        key_columns,
        selected_existing_indices,
        duplicate_summary['staging_table_name'],
    )

    if not selected_file_only_indices:
        return existing_failed_df.drop(columns=['__upload_row_index'], errors='ignore').reset_index(drop=True)

    file_failed_df = df.loc[selected_file_only_indices].copy()
    file_failed_df.insert(0, 'Excel行号', [int(index) + 2 for index in selected_file_only_indices])
    file_failed_df['失败原因'] = 'Excel 内主键或唯一键重复'

    if existing_failed_df.empty:
        return file_failed_df.reset_index(drop=True)

    return pd.concat(
        [
            existing_failed_df.drop(columns=['__upload_row_index'], errors='ignore'),
            file_failed_df,
        ],
        ignore_index=True,
        sort=False,
    )


def _build_insert_duplicate_update_rows(
    db_name: str,
    target_meta: dict,
    df: pd.DataFrame,
    duplicate_summary: dict,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[list[str], list[dict]]:
    existing_indices = set(duplicate_summary.get('existing_row_indices') or [])
    file_indices = set(duplicate_summary.get('file_row_indices') or [])
    key_columns = list(duplicate_summary.get('key_columns') or [])
    target_columns = list(target_meta.get('columns') or [])
    upload_columns = list(df.columns)

    records = []
    existing_row_indices = sorted(existing_indices)
    file_only_indices = sorted(file_indices - existing_indices)
    offset = max(0, int(offset or 0))
    if limit is not None:
        limit = max(0, int(limit))
        existing_count = len(existing_row_indices)
        existing_start = min(offset, existing_count)
        existing_end = min(existing_count, offset + limit)
        selected_existing_indices = existing_row_indices[existing_start:existing_end]
        remaining_limit = max(0, limit - len(selected_existing_indices))
        file_start = max(0, offset - existing_count)
        selected_file_only_indices = file_only_indices[file_start:file_start + remaining_limit]
    else:
        existing_count = len(existing_row_indices)
        selected_existing_indices = existing_row_indices[offset:]
        file_start = max(0, offset - existing_count)
        selected_file_only_indices = file_only_indices[file_start:]

    existing_rows = _fetch_insert_existing_duplicate_rows(
        db_name,
        target_meta,
        df,
        key_columns,
        selected_existing_indices,
        duplicate_summary['staging_table_name'],
    )
    for row in safe_records(existing_rows):
        upload_index = row.get('__upload_row_index')
        if upload_index is None:
            continue
        upload_index = int(upload_index)
        record = {
            '__upload_row_index': upload_index,
            '可更新': True,
            'Excel行号': upload_index + 2,
            '失败原因': row.get('失败原因') or '目标表已存在相同主键或唯一键',
        }
        for column in key_columns:
            record[f'主键:{column}'] = _json_safe_cell(df.at[upload_index, column])
        for column in target_columns:
            record[f'当前:{column}'] = row.get(column)
        for column in upload_columns:
            record[f'上传:{column}'] = _json_safe_cell(df.at[upload_index, column])
        records.append(record)

    for upload_index in selected_file_only_indices:
        record = {
            '__upload_row_index': int(upload_index),
            '可更新': False,
            'Excel行号': int(upload_index) + 2,
            '失败原因': 'Excel 内主键或唯一键重复',
        }
        for column in key_columns:
            record[f'主键:{column}'] = _json_safe_cell(df.at[upload_index, column])
        for column in target_columns:
            record[f'当前:{column}'] = None
        for column in upload_columns:
            record[f'上传:{column}'] = _json_safe_cell(df.at[upload_index, column])
        records.append(record)

    columns = (
        ['可更新', 'Excel行号', '失败原因']
        + [f'主键:{column}' for column in key_columns]
        + [f'当前:{column}' for column in target_columns]
        + [f'上传:{column}' for column in upload_columns]
    )
    return columns, records


def _resolve_insert_auto_update_time_column(
    target_meta: dict,
    upload_columns: list[str],
    key_columns: list[str],
) -> str | None:
    upload_column_set = set(upload_columns)
    key_column_set = set(key_columns)
    target_columns = list(target_meta.get('columns') or [])
    for candidate_name in ('update_time', 'updated_at'):
        for column in target_columns:
            if column.lower() != candidate_name:
                continue
            if column in upload_column_set or column in key_column_set:
                return None
            return column
    return None


def _update_insert_duplicate_rows_from_staging(
    db_name: str,
    target_meta: dict,
    df: pd.DataFrame,
    duplicate_summary: dict,
    selected_rows: list[int] | None = None,
) -> int:
    """用上传批次 staging 表批量更新已确认的历史重复 ID。"""
    staging_table_name = str(
        duplicate_summary.get('staging_table_name') or ''
    ).strip()
    key_columns = list(duplicate_summary.get('key_columns') or [])
    existing_indices = set(duplicate_summary.get('existing_row_indices') or [])
    if not staging_table_name:
        raise ValueError('当前重复数据没有可用的 staging 会话，请重新上传 Excel')
    if not key_columns:
        raise ValueError('没有可更新的历史重复数据')

    selected_indices = sorted(existing_indices)
    selected_filter = 'COALESCE(s."__insert_existing_match", FALSE)'
    selected_params = {}
    if selected_rows is not None:
        selected_indices = []
        for row_index in selected_rows:
            try:
                normalized = int(row_index)
            except (TypeError, ValueError):
                continue
            if normalized not in selected_indices:
                selected_indices.append(normalized)
        if not selected_indices:
            raise ValueError('请选择需要更新的重复数据')
        invalid_indices = [
            index for index in selected_indices
            if index not in existing_indices
        ]
        if invalid_indices:
            raise ValueError('只能更新目标表中已存在的重复 ID，Excel 内部重复行不能直接更新')

        selected_filter += ' AND s."__upload_row_index" IN :selected_indices'
        selected_params['selected_indices'] = selected_indices

    seen_keys = set()
    for row_index in selected_indices:
        key_values = []
        for column in key_columns:
            value = _json_safe_cell(df.at[row_index, column])
            try:
                hash(value)
            except TypeError:
                value = json.dumps(value, ensure_ascii=False, default=str)
            key_values.append(value)
        key_signature = tuple(key_values)
        if key_signature in seen_keys:
            raise ValueError('同一个主键或唯一键只能选择一行进行更新，请先核对 Excel 重复数据')
        seen_keys.add(key_signature)

    update_columns = [
        column for column in df.columns
        if column not in key_columns
    ]
    auto_update_time_column = _resolve_insert_auto_update_time_column(
        target_meta,
        list(df.columns),
        key_columns,
    )
    if not update_columns and not auto_update_time_column:
        raise ValueError('Excel 除主键或唯一键外没有可更新字段')

    target_table_sql = _quote_table_name(target_meta)
    staging_table_sql = _insert_staging_table_sql(
        target_meta,
        staging_table_name,
    )
    source_columns = ', '.join(
        f's.{_quote_ident(column)} AS {_quote_ident(column)}'
        for column in df.columns
    )
    partition_columns = ', '.join(
        f's.{_quote_ident(column)}'
        for column in key_columns
    )
    set_items = [
        f'{_quote_ident(column)} = u.{_quote_ident(column)}'
        for column in update_columns
    ]
    if auto_update_time_column:
        set_items.append(
            f'{_quote_ident(auto_update_time_column)} = CURRENT_TIMESTAMP'
        )
    join_condition = _insert_join_condition('t', 'u', key_columns)
    query = text(f"""
        WITH ranked_upload AS (
            SELECT
                {source_columns},
                ROW_NUMBER() OVER (
                    PARTITION BY {partition_columns}
                    ORDER BY s."__upload_row_index"
                ) AS "__insert_rank"
            FROM {staging_table_sql} AS s
            WHERE {selected_filter}
        )
        UPDATE {target_table_sql} AS t
        SET {', '.join(set_items)}
        FROM ranked_upload AS u
        WHERE u."__insert_rank" = 1
          AND {join_condition}
    """)
    if selected_rows is not None:
        query = query.bindparams(
            bindparam('selected_indices', expanding=True)
        )

    engine = DatabasePoolManager.get_engine(db_name)
    with engine.begin() as conn:
        marked_count = conn.execute(text(f"""
            SELECT COUNT(*)
            FROM {staging_table_sql}
            WHERE COALESCE("__insert_existing_match", FALSE)
        """)).scalar()
        if not marked_count:
            raise ValueError('上传批次没有完成历史重复标记')
        result = conn.execute(query, selected_params)
        rowcount = result.rowcount
    if rowcount and rowcount > 0:
        return int(rowcount)
    # Some PostgreSQL-compatible drivers report -1 for UPDATE rowcount.
    return len(selected_indices)


def _remove_insert_duplicate_indices(duplicate_summary: dict, selected_rows: list[int]):
    selected_indices = set()
    for row_index in selected_rows:
        try:
            selected_indices.add(int(row_index))
        except (TypeError, ValueError):
            continue
    if not selected_indices:
        return

    for key in ('existing_row_indices', 'file_row_indices', 'row_indices'):
        duplicate_summary[key] = [
            int(index) for index in duplicate_summary.get(key, [])
            if int(index) not in selected_indices
        ]
    duplicate_summary['existing_rows'] = len(duplicate_summary.get('existing_row_indices') or [])
    duplicate_summary['file_rows'] = len(duplicate_summary.get('file_row_indices') or [])
    duplicate_summary['total_rows'] = len(duplicate_summary.get('row_indices') or [])


def _guess_match_field(columns: list[str], keyword_column: str, hint_text: str = '') -> str | None:
    if not columns:
        return None

    keyword_norm = _clean_name(keyword_column)
    hint_norm = _clean_name(hint_text)
    hints = ('keyword', 'name', 'title', 'code', 'id', 'key', 'term', 'value', 'desc', 'text', 'label', 'content', 'alias')
    best_score = -1
    best_col = None

    for col in columns:
        col_norm = _clean_name(col)
        score = 0
        if col_norm and col_norm == keyword_norm:
            score += 100
        if keyword_norm and (keyword_norm in col_norm or col_norm in keyword_norm):
            score += 40
        if hint_norm and (hint_norm in col_norm or col_norm in hint_norm):
            score += 20
        for hint in hints:
            if hint in col_norm:
                score += 10
        if col_norm.endswith(('name', 'code', 'id', 'key')):
            score += 5
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def _guess_return_fields(columns: list[str], match_field: str, hint_text: str = '', limit: int = 5) -> list[str]:
    if not columns:
        return []

    hint_norm = _clean_name(hint_text)
    hints = ('name', 'title', 'code', 'id', 'category', 'type', 'brand', 'sku', 'desc', 'description', 'status', 'value', 'label')
    scored = []
    for col in columns:
        if col == match_field:
            continue
        col_norm = _clean_name(col)
        score = 0
        if hint_norm and (hint_norm in col_norm or col_norm in hint_norm):
            score += 30
        for hint in hints:
            if hint in col_norm:
                score += 10
        if col_norm.endswith(('name', 'code', 'id', 'title')):
            score += 5
        scored.append((score, col))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [col for _, col in scored[:limit]]
    return selected


def _guess_keyword_column(df: pd.DataFrame, hint_text: str = '') -> str | None:
    if df is None or df.empty:
        return None

    hint_norm = _clean_name(hint_text)
    hints = ('keyword', 'name', 'title', 'code', 'id', 'key', 'term', 'value', 'desc', 'text', 'label', 'content', 'alias', '商品', '客户', '订单')
    best_score = -10**9
    best_col = None

    for col in df.columns:
        col_norm = _clean_name(col)
        score = 0
        if hint_norm and (hint_norm in col_norm or col_norm in hint_norm):
            score += 20
        for hint in hints:
            if _clean_name(hint) in col_norm:
                score += 8
        dtype = df[col].dtype
        if pd.api.types.is_string_dtype(dtype) or dtype == object:
            score += 5
        elif pd.api.types.is_numeric_dtype(dtype):
            score -= 3
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def _sample_keyword_values(df: pd.DataFrame, keyword_column: str, limit: int = 5) -> list[str]:
    if keyword_column not in df.columns:
        return []

    samples = []
    for value in df[keyword_column].tolist():
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass

        text = str(value).strip()
        if not text or text in samples:
            continue
        samples.append(text)
        if len(samples) >= limit:
            break
    return samples


def _guess_keyword_column_from_columns(columns: list[str], hint_text: str = '') -> str | None:
    if not columns:
        return None

    hint_norm = _clean_name(hint_text)
    hints = ('keyword', 'name', 'title', 'code', 'id', 'key', 'term', 'value', 'desc', 'text', 'label', 'content', 'alias', '商品', '客户', '订单')
    best_score = -10**9
    best_col = None

    for col in columns:
        col_norm = _clean_name(col)
        score = 0
        if hint_norm and (hint_norm in col_norm or col_norm in hint_norm):
            score += 20
        for hint in hints:
            if _clean_name(hint) in col_norm:
                score += 8
        if col_norm.endswith(('name', 'code', 'id', 'key')):
            score += 5
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def _extract_upload_base_name(table_name: str) -> str:
    raw = str(table_name or '').strip()
    if raw.endswith('_matched'):
        raw = raw[:-8]
    parsed = _parse_upload_table_name(raw)
    if parsed:
        return parsed['base_name']
    return raw


def _resolve_upload_match_plan(
    db_name: str,
    source_table_name: str,
    source_columns: list[str],
    keyword_column: str,
    match_hint: str,
    *,
    template_key: str = '',
    match_table_input: str = '',
    match_field_input: str = '',
    field_mappings: list[dict] | None = None,
    return_fields_raw: str = '',
    match_mode: str = 'exact',
    schema_filter: str | None = None,
    auto_match_table: bool = True,
    use_template_mode: bool = True,
    ai_mode: bool = False,
    sample_keywords: list[str] | None = None,
    match_config: dict | None = None,
) -> dict:
    field_mappings = _normalize_match_field_mappings(
        field_mappings,
        keyword_column,
        match_field_input,
    )
    tables = KnowledgeBase(db_name).get_table_list()
    upload_match_config = match_config if isinstance(match_config, dict) else get_upload_match_config_from_db(
        db_name,
        template_key=template_key if use_template_mode else None,
    )

    configured_match_table = str(upload_match_config.get('match_table', '') or '').strip()
    configured_match_field = str(upload_match_config.get('match_field', '') or '').strip()
    configured_match_mode = str(upload_match_config.get('match_mode', '') or '').strip().lower()
    configured_return_fields = upload_match_config.get('return_fields') if 'return_fields' in upload_match_config else None
    configured_target_filter = str(upload_match_config.get('target_filter', '') or '').strip()
    configured_sql_text = str(upload_match_config.get('sql_text', '') or '').strip()
    target_sql_text = _normalize_template_select_sql(configured_sql_text) if use_template_mode and configured_sql_text else ''

    target_table_meta = None
    if use_template_mode and configured_match_table:
        target_table_meta = _resolve_table_meta(tables, configured_match_table)
        if not target_table_meta:
            raise ValueError(f'配置中的目标表不存在: {configured_match_table}')
    elif match_table_input:
        target_table_meta = _resolve_table_meta(tables, match_table_input)
        if not target_table_meta:
            raise ValueError(f'未找到目标表: {match_table_input}')

    if not target_table_meta:
        search_parts = [match_hint, keyword_column]
        for mapping in field_mappings:
            search_parts.extend([
                mapping.get('source_field') or '',
                mapping.get('target_field') or '',
            ])
        if sample_keywords:
            search_parts.extend(sample_keywords)
        search_query = ' '.join(part for part in search_parts if part).strip()
        if search_query and (auto_match_table or ai_mode or template_key):
            search_results = TableSchemaSearcher.search(
                db_name,
                search_query,
                top_k=3,
                schema_filter=schema_filter,
            )
            if search_results:
                suggested = search_results[0]
                suggested_name = f"{suggested.get('schema')}.{suggested.get('table_name')}" if suggested.get('schema') else suggested.get('table_name')
                target_table_meta = _resolve_table_meta(tables, suggested_name)

    if not target_table_meta:
        raise ValueError('未能识别目标表，请在模板中配置，或补充业务描述')

    target_columns = target_table_meta.get('columns') or []
    if not target_columns:
        raise ValueError(f"目标表 {_format_full_table_name(target_table_meta)} 没有可用字段")

    explicit_match_fields = [
        str(mapping.get('target_field') or '').strip()
        for mapping in field_mappings
        if str(mapping.get('target_field') or '').strip()
    ]
    if explicit_match_fields:
        raw_match_field = explicit_match_fields[0]
    elif match_field_input:
        raw_match_field = match_field_input
    elif use_template_mode and configured_match_field:
        raw_match_field = configured_match_field
    else:
        raw_match_field = _guess_match_field(target_columns, keyword_column, match_hint)
    if not raw_match_field:
        raise ValueError('未能识别匹配字段，请在模板中配置，或补充业务描述')

    if use_template_mode and configured_return_fields is not None:
        return_field_specs = _normalize_return_field_specs(configured_return_fields)
    elif return_fields_raw:
        return_field_specs = _normalize_return_field_specs(return_fields_raw)
    elif ai_mode:
        return_field_specs = [
            {
                'db_field': field,
                'business_field': field,
                'enabled': True,
            }
            for field in _guess_return_fields(target_columns, raw_match_field, match_hint or source_table_name, limit=5)
        ]
    else:
        return_field_specs = [
            {
                'db_field': field,
                'business_field': field,
                'enabled': True,
            }
            for field in target_columns if field != raw_match_field
        ][:5]

    def resolve_match_field_spec(field_name: str) -> dict | None:
        return _resolve_target_field_spec(field_name, return_field_specs, target_columns)

    match_field_spec = resolve_match_field_spec(raw_match_field)
    if not match_field_spec:
        raise ValueError(f'匹配字段 {raw_match_field} 不在目标表字段或返回字段映射中')

    def format_match_pair(source_field: str, spec: dict, raw_target: str = '') -> dict:
        target_label = str(spec.get('business_field') or spec.get('db_field') or raw_target).strip()
        return {
            'source_field': str(source_field or '').strip().replace(' ', '_'),
            'target_field': target_label,
            'match_field_spec': spec,
        }

    match_field_pairs: list[dict] = []
    source_mappings = field_mappings or [{'source_field': keyword_column, 'target_field': raw_match_field}]
    for index, mapping in enumerate(source_mappings):
        source_field = str(mapping.get('source_field') or '').strip().replace(' ', '_')
        raw_target_field = str(mapping.get('target_field') or '').strip()
        if not source_field and index == 0:
            source_field = keyword_column
        if not source_field:
            raise ValueError(f'第 {index + 1} 组匹配缺少目标表字段')

        pair_spec = None
        if raw_target_field:
            pair_spec = resolve_match_field_spec(raw_target_field)
            if not pair_spec:
                raise ValueError(f'匹配字段 {raw_target_field} 不在目标表字段或返回字段映射中')
        elif index == 0:
            pair_spec = match_field_spec
        elif ai_mode or not use_template_mode or not template_key:
            guessed_field = _guess_match_field(target_columns, source_field, match_hint)
            if guessed_field:
                pair_spec = resolve_match_field_spec(guessed_field)

        if not pair_spec:
            raise ValueError(f'第 {index + 1} 组匹配缺少模版字段')

        match_field_pairs.append(format_match_pair(source_field, pair_spec, raw_target_field))

    match_field_label = str(match_field_spec.get('business_field') or match_field_spec.get('db_field') or raw_match_field).strip()
    match_field_db_values = {
        str(pair['match_field_spec'].get('db_field') or pair.get('target_field') or '').strip()
        for pair in match_field_pairs
        if pair.get('match_field_spec')
    }

    return_field_specs = [spec for spec in return_field_specs if spec.get('enabled', True)]
    if return_field_specs:
        invalid_fields = [
            spec.get('db_field')
            for spec in return_field_specs
            if _is_simple_identifier(spec.get('db_field', '')) and spec.get('db_field') not in target_columns
        ]
        invalid_fields = [field for field in invalid_fields if field]
        if invalid_fields:
            raise ValueError(f"返回字段不存在于目标表中: {', '.join(sorted(set(invalid_fields)))}")
    else:
        return_field_specs = [
            {
                'db_field': field,
                'business_field': field,
                'enabled': True,
            }
            for field in target_columns if field not in match_field_db_values
        ][:5]

    if configured_match_mode in {'exact', 'contains'} and use_template_mode:
        match_mode = configured_match_mode
    if match_mode not in {'exact', 'contains'}:
        match_mode = 'exact'

    return {
        'target_table_meta': target_table_meta,
        'match_field_spec': match_field_spec,
        'match_field': match_field_label,
        'match_field_pairs': match_field_pairs,
        'field_mappings': [
            {
                'source_field': pair.get('source_field') or '',
                'target_field': pair.get('target_field') or '',
            }
            for pair in match_field_pairs
        ],
        'match_mode': match_mode,
        'configured_target_filter': configured_target_filter,
        'target_sql_text': target_sql_text,
        'return_field_specs': return_field_specs,
    }


def _execute_match_query(
    db_name: str,
    source_table_name: str,
    matched_table_name: str,
    keyword_column: str,
    target_table_meta: dict,
    match_field_spec: dict,
    return_field_specs: list[dict],
    match_mode: str,
    target_filter: str = '',
    source_columns: list[str] | None = None,
    match_field_pairs: list[dict] | None = None,
    target_sql_text: str = '',
) -> dict:
    engine = DatabasePoolManager.get_engine(db_name)
    matched_table_sql = _quote_ident(matched_table_name)
    match_query = _build_match_sql(
        source_table_name,
        keyword_column,
        target_table_meta,
        match_field_spec,
        return_field_specs,
        match_mode,
        target_filter,
        source_columns,
        match_field_pairs,
        target_sql_text,
    )
    result_query = _build_match_result_sql(matched_table_name)

    with engine.connect() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS tmp.{matched_table_sql} CASCADE'))
        conn.execute(text(f'CREATE TABLE tmp.{matched_table_sql} AS {match_query}'))
        conn.commit()

        result_rows = conn.execute(text(f"""
            SELECT COUNT(*)
            FROM tmp.{matched_table_sql}
        """)).fetchone()[0]
        matched_rows = conn.execute(text(f"""
            SELECT COUNT(*)
            FROM tmp.{matched_table_sql}
            WHERE match_hit IS TRUE
        """)).fetchone()[0]

        preview_df = pd.read_sql(text(f'{result_query} LIMIT 200'), conn)

    return {
        'match_query': match_query,
        'result_query': result_query,
        'result_rows': int(result_rows or 0),
        'matched_rows': int(matched_rows or 0),
        'preview_df': preview_df,
    }


def _build_match_sql(
    source_table_name: str,
    keyword_column: str,
    target_table_meta: dict,
    match_field_spec: dict,
    return_field_specs: list[dict],
    match_mode: str,
    target_filter: str = '',
    source_columns: list[str] | None = None,
    match_field_pairs: list[dict] | None = None,
    target_sql_text: str = '',
) -> str:
    target_columns = target_table_meta.get('columns') or []
    template_query_sql = _normalize_template_select_sql(target_sql_text) if target_sql_text else ''
    use_template_query = bool(template_query_sql)
    normalized_pairs = []
    for pair in match_field_pairs or []:
        if not isinstance(pair, dict):
            continue
        source_field = str(pair.get('source_field') or '').strip().replace(' ', '_')
        pair_spec = pair.get('match_field_spec')
        if source_field and isinstance(pair_spec, dict):
            normalized_pairs.append({
                'source_field': source_field,
                'match_field_spec': pair_spec,
            })

    if not normalized_pairs and keyword_column and isinstance(match_field_spec, dict):
        normalized_pairs.append({
            'source_field': keyword_column,
            'match_field_spec': match_field_spec,
        })

    match_specs, select_specs, match_alias_keys = _build_match_target_select_specs(
        [pair['match_field_spec'] for pair in normalized_pairs],
        return_field_specs,
        target_columns,
        use_template_query=use_template_query,
    )
    if not match_specs or not normalized_pairs:
        raise ValueError('匹配字段不在目标表字段或返回字段映射中')

    used_aliases = {_clean_name(column) for column in (source_columns or []) if str(column or '').strip()}
    final_alias_by_key: dict[str, str] = {}
    resolved_select_specs: list[dict] = []
    for spec in select_specs:
        base_alias = str(spec.get('business_field') or spec.get('db_field') or '').strip()
        if not base_alias:
            continue
        alias = base_alias
        suffix = 2
        alias_key = _clean_name(alias)
        while alias_key in used_aliases:
            alias = f'{base_alias}_{suffix}'
            alias_key = _clean_name(alias)
            suffix += 1
        used_aliases.add(alias_key)
        original_alias_key = str(spec.get('alias_key') or _clean_name(base_alias))
        final_alias_by_key[original_alias_key] = alias
        resolved_select_specs.append({**spec, 'alias': alias})

    if not resolved_select_specs:
        raise ValueError('未能生成可用的匹配字段')

    target_sql = f'({template_query_sql})' if use_template_query else _quote_table_name(target_table_meta)
    source_sql = f'tmp.{_quote_ident(source_table_name)}'
    join_parts = []
    for index, pair in enumerate(normalized_pairs):
        alias_key = match_alias_keys.get(index)
        match_alias = final_alias_by_key.get(alias_key or '')
        if not match_alias:
            raise ValueError('匹配字段缺少有效名称')

        keyword_ident = _quote_ident(pair['source_field'])
        match_ident = _quote_ident(match_alias)
        source_norm = f"LOWER(TRIM(COALESCE(CAST(a.{keyword_ident} AS TEXT), '')))"
        target_norm = f"LOWER(TRIM(COALESCE(CAST(b.{match_ident} AS TEXT), '')))"
        non_empty = f"{source_norm} <> '' AND {target_norm} <> ''"
        if match_mode == 'contains':
            join_parts.append(
                f"({non_empty} AND ({target_norm} LIKE '%' || {source_norm} || '%' "
                f"OR {source_norm} LIKE '%' || {target_norm} || '%'))"
            )
        else:
            join_parts.append(f"({non_empty} AND {target_norm} = {source_norm})")

    join_clause = ' AND '.join(join_parts)

    validated_filter = '' if use_template_query else _validate_target_filter(target_filter)
    if validated_filter:
        validated_filter = _rewrite_target_column_qualifiers(
            validated_filter,
            target_columns,
        )
        validated_filter = f'WHERE {validated_filter}'
    else:
        validated_filter = ''

    target_parts = ['TRUE AS __matched']
    for spec in resolved_select_specs:
        alias = str(spec.get('alias') or '').strip()
        if not alias:
            continue
        target_parts.append(
            f'{spec.get("expression") or "NULL"} AS {_quote_ident(alias)}'
        )

    outer_parts = ['a.*', 'COALESCE(b.__matched, FALSE) AS match_hit']
    for spec in resolved_select_specs:
        alias = str(spec.get('alias') or '').strip()
        if alias:
            outer_parts.append(f'b.{_quote_ident(alias)}')

    return f"""
        SELECT {', '.join(outer_parts)}
        FROM {source_sql} a
        LEFT JOIN (
            SELECT {', '.join(target_parts)}
            FROM {target_sql} t
            {validated_filter}
        ) b ON {join_clause}
    """


def _build_match_result_sql(
    matched_table_name: str,
    matched_only: bool = False,
) -> str:
    """Build the user-facing query for a materialized match result."""
    sql = f'SELECT * FROM tmp.{_quote_ident(matched_table_name)}'
    if matched_only:
        sql += ' WHERE match_hit IS TRUE'
    return sql


def _parse_match_history_payload(raw_value) -> dict:
    if isinstance(raw_value, dict):
        return raw_value
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _display_upload_template_actor(user: dict | None = None) -> str:
    if not user:
        return ''
    for key in ('user_name', 'mobile', 'we_user_id', 'employee_id', 'admin_id'):
        value = user.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ''


def _current_upload_template_actor(payload: dict | None = None) -> str:
    payload = payload or {}
    for key in ('updated_by', 'created_by', 'operator', 'user_name', 'username'):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    current_user = get_current_user()
    actor = _display_upload_template_actor(current_user)
    if actor:
        return actor
    for header_name in ('X-User', 'X-Username', 'X-Operator'):
        value = request.headers.get(header_name, '').strip()
        if value:
            return value
    return 'system'


def _format_upload_template_timestamp() -> str:
    return datetime.now().isoformat(timespec='seconds')


def _detect_duplicate_upload_template_label(
    db_config: dict,
    template_key: str,
    label: str,
) -> list[str]:
    normalized = normalize_upload_template_label(label)
    if not normalized:
        return []

    duplicates = []
    for key, config in (db_config or {}).items():
        if key == template_key or not isinstance(config, dict):
            continue
        other_label = normalize_upload_template_label(config.get('label') or config.get('name') or key)
        if other_label and other_label == normalized:
            duplicates.append(key)
    return duplicates


def _merge_upload_template_config(
    template_key: str,
    incoming_config: dict,
    existing_config: dict | None = None,
    actor: str = 'system',
) -> dict:
    existing_config = dict(existing_config or {})
    merged = dict(existing_config)
    merged.update(incoming_config or {})

    def merged_text(field: str, default: str = '') -> str:
        incoming_value = str((incoming_config or {}).get(field) or '').strip()
        existing_value = str(existing_config.get(field) or '').strip()
        return incoming_value or existing_value or default

    label = str(merged.get('label') or merged.get('name') or template_key or '').strip()
    merged['label'] = label or template_key
    merged['name'] = merged.get('name') or merged['label']
    merged['match_table'] = merged_text('match_table')
    merged['match_field'] = merged_text('match_field')
    merged['match_field_display'] = merged_text('match_field_display', merged['match_field'])
    merged['match_mode'] = merged_text('match_mode', 'exact') or 'exact'
    merged['target_filter'] = merged_text('target_filter')
    merged['sql_text'] = merged_text('sql_text')
    merged['description'] = str(merged.get('description') or '').strip()
    merged['keyword_column'] = str(merged.get('keyword_column') or '').strip()
    incoming_has_return_fields = isinstance(incoming_config, dict) and 'return_fields' in incoming_config
    incoming_return_fields = _normalize_return_field_specs(
        incoming_config.get('return_fields') if incoming_has_return_fields else None
    )
    existing_return_fields = _normalize_return_field_specs(existing_config.get('return_fields'))
    if incoming_has_return_fields:
        merged['return_fields'] = incoming_return_fields or existing_return_fields
    else:
        merged['return_fields'] = existing_return_fields

    now = _format_upload_template_timestamp()
    merged['created_by'] = str(existing_config.get('created_by') or merged.get('created_by') or actor).strip() or 'system'
    merged['created_at'] = str(existing_config.get('created_at') or merged.get('created_at') or now).strip() or now
    merged['updated_by'] = actor
    merged['updated_at'] = now

    return merged


def _safe_next_url(raw_next: str) -> str:
    next_url = (raw_next or '').strip()
    if not next_url or not next_url.startswith('/') or next_url.startswith('//'):
        return url_for('main.index')
    return next_url


@main_bp.route('/login', methods=['GET'])
def login_page():
    """登录页"""
    if get_current_user():
        return redirect(_safe_next_url(request.args.get('next', '')))
    return render_template('login.html')


@main_bp.route('/api/login', methods=['POST'])
def login():
    """账号密码登录"""
    data = request.get_json(silent=True) or {}
    login_name = (data.get('login_name') or data.get('account') or '').strip()
    password = data.get('password') or ''
    if not login_name or not password:
        return jsonify({'status': 'error', 'error': '请输入账号和密码'}), 400

    try:
        user = authenticate_user(login_name, password)
    except Exception as e:
        return jsonify({'status': 'error', 'error': f'登录失败: {e}'}), 500

    if not user:
        return jsonify({'status': 'error', 'error': '账号或密码错误'}), 401

    login_user(user)
    return jsonify({'status': 'success', 'user': public_user_payload(user)})


@main_bp.route('/api/logout', methods=['POST'])
def logout():
    """退出登录"""
    logout_user()
    return jsonify({'status': 'success'})


@main_bp.route('/api/current_user', methods=['GET'])
def current_user():
    """当前登录用户"""
    user = get_current_user()
    if not user:
        return jsonify({'status': 'unauthorized', 'logged_in': False}), 401
    return jsonify({'status': 'success', 'logged_in': True, 'user': public_user_payload(user)})


@main_bp.route('/api/admin/sync_app_users', methods=['POST'])
def sync_app_users():
    """同步账号资料和密码。"""
    current_user_data = get_current_user()
    if not user_can_view_all(current_user_data):
        return jsonify({'status': 'forbidden', 'error': '仅管理员可执行同步'}), 403

    result = sync_app_users_from_source()
    status = 'success' if result.get('ok') else 'error'
    code = 200 if result.get('ok') else 500
    return jsonify({'status': status, 'result': result}), code


@main_bp.route('/api/admin/sync_book', methods=['POST'])
def sync_book():
    """手动同步仓库代码知识。"""
    current_user_data = get_current_user()
    if not user_can_view_all(current_user_data):
        return jsonify({'status': 'forbidden', 'error': '仅管理员可执行同步'}), 403

    result = sync_book_from_source(force=True)
    status = 'success' if result.get('ok') else 'error'
    code = 200 if result.get('ok') else 500
    return jsonify({'status': status, 'result': result}), code


@main_bp.route('/api/admin/sync_book_status', methods=['GET'])
def sync_book_status():
    """查看仓库代码同步状态。"""
    current_user_data = get_current_user()
    if not user_can_view_all(current_user_data):
        return jsonify({'status': 'forbidden', 'error': '仅管理员可查看状态'}), 403

    return jsonify({
        'status': 'success',
        'running': is_book_sync_monitor_running(),
        'result': get_last_book_sync_result(),
    })


@main_bp.route('/')
def index():
    """主页"""
    return render_template('index.html', current_user=public_user_payload(get_current_user()))


@main_bp.route('/get_databases', methods=['GET'])
def get_databases():
    """获取可用数据库列表"""
    try:
        databases = get_available_databases()
        return jsonify({'databases': databases, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': _format_match_error(e), 'status': 'error'}), 500


@main_bp.route('/api/db_tables', methods=['GET'])
def get_db_tables():
    """获取数据库表列表"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    schema_filter = request.args.get('schema_name', '').strip() or None
    schemas = [s.strip() for s in (schema_filter or '').split(',') if s.strip()]
    include_columns = request.args.get('include_columns', '1').strip().lower() not in {'0', 'false', 'no'}
    try:
        try:
            live_tables = _get_live_database_tables(db_name, schemas or None, include_columns=include_columns)
            if live_tables:
                tables = sorted(
                    live_tables,
                    key=lambda table: (str(table.get('schema') or ''), str(table.get('table_name') or '')),
                )
                return jsonify({'status': 'success', 'tables': tables})
        except Exception as live_error:
            print(f'实时表结构读取失败，使用知识库缓存: {live_error}')

        cached_tables = KnowledgeBase(db_name).get_table_list()
        merged = {}
        for table in cached_tables:
            key = (str(table.get('schema') or ''), str(table.get('table_name') or ''))
            if key[1]:
                item = dict(table)
                if not include_columns:
                    item['columns'] = []
                    item['column_count'] = 0
                    item['columns_loaded'] = False
                else:
                    item['columns_loaded'] = True
                merged[key] = item
        tables = sorted(
            merged.values(),
            key=lambda table: (str(table.get('schema') or ''), str(table.get('table_name') or '')),
        )
        if schemas:
            schema_set = set(schemas)
            tables = [t for t in tables if t.get('schema') in schema_set]
        return jsonify({'status': 'success', 'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/db_schemas', methods=['GET'])
def get_db_schemas():
    """获取数据库的 schema 列表"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    try:
        try:
            schemas = _get_live_database_schemas(db_name)
            if schemas:
                return jsonify({'status': 'success', 'schemas': schemas})
        except Exception as live_error:
            print(f'实时 schema 读取失败，使用知识库缓存: {live_error}')

        tables = KnowledgeBase(db_name).get_table_list()
        schemas = sorted(set(t['schema'] for t in tables if t.get('schema')))
        return jsonify({'status': 'success', 'schemas': schemas})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/insert_target_tables', methods=['GET'])
def insert_target_tables():
    """查询可用于插入的 tmp schema 表。"""
    db_name = request.args.get('db_name', '').strip()
    keyword = request.args.get('keyword', '').strip().lower()
    try:
        limit = int(request.args.get('limit', 50) or 50)
    except Exception:
        limit = 50
    limit = max(1, min(limit, 200))
    if not db_name:
        return jsonify({'error': 'missing db_name', 'status': 'error'}), 400

    try:
        keyword_like = f'%{keyword}%'
        query = text("""
            SELECT table_schema, table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = 'tmp'
              AND lower(table_name) NOT LIKE '%\\_middle%' ESCAPE '\\'
              AND lower(table_name) NOT LIKE '\\_\\_insert\\_stage\\_%' ESCAPE '\\'
              AND (
                  :keyword = ''
                  OR lower(table_name) LIKE :keyword_like
                  OR lower(table_schema || '.' || table_name) LIKE :keyword_like
              )
            ORDER BY table_name
            LIMIT :limit
        """)
        engine = DatabasePoolManager.get_engine(db_name)
        with engine.connect() as conn:
            rows = conn.execute(
                query,
                {
                    'keyword': keyword,
                    'keyword_like': keyword_like,
                    'limit': limit,
                },
            ).fetchall()
        tables = []
        for row in rows:
            mapping = row._mapping
            table_name = str(mapping.get('table_name') or '').strip()
            if not table_name:
                continue
            tables.append({
                'schema': 'tmp',
                'table_name': table_name,
                'table_type': str(mapping.get('table_type') or '').strip(),
                'columns': [],
                'column_count': 0,
                'columns_loaded': False,
            })
        return jsonify({
            'status': 'success',
            'schema': 'tmp',
            'keyword': keyword,
            'tables': tables,
        })
    except Exception as e:
        return jsonify({'error': _format_match_error(e), 'status': 'error'}), 500


@main_bp.route('/api/knowledge/status', methods=['GET'])
def knowledge_status():
    """获取知识库状态"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    
    kb = SQLKnowledgeRepo(db_name, current_user=get_current_user())
    knowledge = kb.list()
    
    return jsonify({
        'status': 'success',
        'db_name': db_name,
        'row_count': len(knowledge),
        'from_upload': False
    })


@main_bp.route('/execute_sql', methods=['POST'])
def execute_sql():
    """执行SQL查询"""
    start_time = time.time()
    request_id = ''
    cancel_token = None
    try:
        data = request.get_json() or {}
        sql = data.get('sql')
        db_name = data.get('db_name')
        client_request_id = (data.get('request_id') or '').strip() if isinstance(data.get('request_id'), str) else ''
        if not sql or not db_name:
            return jsonify({'error': 'Missing sql or db_name'}), 400

        if client_request_id:
            cancel_token = cancel_registry.register_with_id(client_request_id)
            request_id = client_request_id
        else:
            request_id, cancel_token = cancel_registry.create()

        db = DatabaseManager(db_name)
        result = db.execute_sql(sql, cancel_token=cancel_token)
        records = safe_records(result)
        total_duration_ms = (time.time() - start_time) * 1000

        insert_query_log(db_name, {
            'db_name': db_name,
            'nl_query': sql,
            'schema_filter': None,
            'search_mode': 'manual_sql',
            'selected_table': None,
            'top_k': None,
            'matched_tables': None,
            'generated_sql': sql,
            'execute_status': 'success',
            'error_message': None,
            'result_rows': len(result),
            'search_duration_ms': 0,
            'llm_duration_ms': 0,
            'sql_exec_duration_ms': total_duration_ms,
            'total_duration_ms': total_duration_ms,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'llm_calls': 0,
            'request_id': request_id,
            'action_type': 'manual_sql',
        })

        return jsonify({'sql_result': records, 'columns': list(result.columns), 'status': 'success', 'request_id': request_id})
    except CancelledError:
        total_duration_ms = (time.time() - start_time) * 1000
        insert_query_log(db_name, {
                'db_name': db_name,
                'nl_query': sql,
                'schema_filter': None,
                'search_mode': 'manual_sql',
                'selected_table': None,
                'top_k': None,
                'matched_tables': None,
                'generated_sql': sql,
                'execute_status': 'cancelled',
                'error_message': '查询已取消',
                'result_rows': 0,
                'search_duration_ms': 0,
                'llm_duration_ms': 0,
                'sql_exec_duration_ms': total_duration_ms,
                'total_duration_ms': total_duration_ms,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'llm_calls': 0,
                'request_id': request_id,
                'action_type': 'manual_sql',
        })
        return jsonify({'error': '查询已取消', 'status': 'cancelled', 'request_id': request_id}), 499
    except Exception as e:
        total_duration_ms = (time.time() - start_time) * 1000
        insert_query_log(db_name, {
                'db_name': db_name,
                'nl_query': sql,
                'schema_filter': None,
                'search_mode': 'manual_sql',
                'selected_table': None,
                'top_k': None,
                'matched_tables': None,
                'generated_sql': sql,
                'execute_status': 'failed',
                'error_message': str(e)[:1000],
                'result_rows': 0,
                'search_duration_ms': 0,
                'llm_duration_ms': 0,
                'sql_exec_duration_ms': total_duration_ms,
                'total_duration_ms': total_duration_ms,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'llm_calls': 0,
                'request_id': request_id,
                'action_type': 'manual_sql',
        })
        return jsonify({'error': str(e), 'status': 'error', 'request_id': request_id}), 500
    finally:
        if request_id:
            cancel_registry.cleanup(request_id)


@main_bp.route('/api/export_query_csv', methods=['POST'])
def export_query_csv():
    """把当前 SQL 查询结果按流式 CSV 导出，避免前端持有全量数据"""
    data = request.get_json(silent=True) or request.form or {}
    db_name = (data.get('db_name') or '').strip()
    sql = (data.get('sql') or '').strip()
    filename = (data.get('filename') or '').strip()

    if not db_name:
        return jsonify({'error': 'missing db_name', 'status': 'error'}), 400
    if not sql:
        return jsonify({'error': 'missing sql', 'status': 'error'}), 400

    cleaned_sql = clean_sql(sql)
    if not cleaned_sql:
        return jsonify({'error': 'SQL 不能为空', 'status': 'error'}), 400

    try:
        validate_sql_safety(cleaned_sql)
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

    export_name = filename or f'query_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    if not export_name.lower().endswith('.csv'):
        export_name += '.csv'
    fallback_name = secure_filename(export_name) or f'query_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    quoted_name = quote(export_name)

    db = DatabaseManager(db_name)

    def normalize_csv_value(value):
        if value is None:
            return ''
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        try:
            if pd.isna(value):
                return ''
        except Exception:
            pass
        if hasattr(value, 'isoformat'):
            try:
                return value.isoformat(sep=' ', timespec='seconds')
            except TypeError:
                try:
                    return value.isoformat()
                except Exception:
                    pass
        return value

    def generate():
        buffer = StringIO()
        writer = csv.writer(buffer)
        buffer.write('\ufeff')

        with db.engine.connect() as conn:
            if db_name == 'hologres':
                try:
                    conn.execute(text("SET hg_computing_resource = 'serverless'"))
                except Exception:
                    pass

            result = conn.execution_options(stream_results=True).execute(text(cleaned_sql))
            writer.writerow(list(result.keys()))
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)

            while True:
                rows = result.fetchmany(1000)
                if not rows:
                    break
                for row in rows:
                    writer.writerow([normalize_csv_value(value) for value in row])
                yield buffer.getvalue()
                buffer.seek(0)
                buffer.truncate(0)

    response = Response(stream_with_context(generate()), mimetype='text/csv; charset=utf-8')
    response.headers['Content-Disposition'] = (
        f"attachment; filename=\"{fallback_name}\"; filename*=UTF-8''{quoted_name}"
    )
    return response


@main_bp.route('/api/query_history', methods=['POST'])
def query_history():
    """获取查询历史记录（支持时间范围筛选）"""
    try:
        data = request.get_json() or {}
        db_name = data.get('db_name')
        limit = min(int(data.get('limit', 50)), 200)
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        search_mode = data.get('search_mode')
        if not db_name:
            return jsonify({'error': 'missing db_name'}), 400

        current_user_data = get_current_user()
        if not current_user_data:
            return jsonify({'error': '请先登录', 'status': 'unauthorized'}), 401

        engine = DatabasePoolManager.get_engine(db_name)

        # 动态构建 WHERE 条件
        where_clauses = ['db_name = :db_name']
        params = {'db_name': db_name, 'limit': limit}
        if not user_can_view_all(current_user_data):
            where_clauses.append('admin_id = :current_admin_id')
            params['current_admin_id'] = current_user_data.get('admin_id')

        if start_time:
            where_clauses.append('created_at >= :start_time')
            params['start_time'] = start_time
        if end_time:
            end_dt = datetime.strptime(end_time, '%Y-%m-%d') + timedelta(days=1)
            where_clauses.append('created_at < :end_time_plus')
            params['end_time_plus'] = end_dt.strftime('%Y-%m-%d')
        if search_mode in ('manual_sql', 'nl_query'):
            where_clauses.append('action_type = :action_type')
            params['action_type'] = search_mode
        elif search_mode:
            where_clauses.append('search_mode = :search_mode')
            params['search_mode'] = search_mode
        else:
            where_clauses.append("(search_mode IS NULL OR search_mode <> 'upload_match')")

        where_sql = ' AND '.join(where_clauses)

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT nl_query, search_mode, generated_sql, execute_status,
                       result_rows, total_duration_ms, error_message, created_at,
                       admin_id, request_id, session_id, client_ip, user_agent,
                       action_type, visibility_scope
                FROM knowledge.query_log
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT :limit
            """), params).fetchall()

        history = []
        for row in rows:
            history.append({
                'nl_query': row._mapping.get('nl_query'),
                'search_mode': row._mapping.get('search_mode'),
                'generated_sql': row._mapping.get('generated_sql'),
                'execute_status': row._mapping.get('execute_status'),
                'result_rows': row._mapping.get('result_rows'),
                'total_duration_ms': row._mapping.get('total_duration_ms'),
                'error_message': row._mapping.get('error_message'),
                'created_at': str(row._mapping.get('created_at')) if row._mapping.get('created_at') else None,
                'admin_id': row._mapping.get('admin_id'),
                'request_id': row._mapping.get('request_id'),
                'session_id': row._mapping.get('session_id'),
                'client_ip': row._mapping.get('client_ip'),
                'user_agent': row._mapping.get('user_agent'),
                'action_type': row._mapping.get('action_type'),
                'visibility_scope': row._mapping.get('visibility_scope'),
            })
        return jsonify({'history': history, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/upload_match_history', methods=['GET'])
def upload_match_history():
    """获取上传表匹配历史，供左侧匹配历史列表展示"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name', 'status': 'error'}), 400

    try:
        limit = int(request.args.get('limit', 30) or 30)
    except Exception:
        limit = 30
    limit = max(1, min(limit, 100))
    start_filter = request.args.get('start_time', '').strip()
    end_filter = request.args.get('end_time', '').strip()
    workflow_mode_filter = request.args.get('workflow_mode', '').strip().lower()
    if workflow_mode_filter and workflow_mode_filter not in {'template', 'ai'}:
        return jsonify({'error': 'invalid workflow_mode', 'status': 'error'}), 400
    for filter_name, filter_value in (('start_time', start_filter), ('end_time', end_filter)):
        if not filter_value:
            continue
        try:
            datetime.strptime(filter_value, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': f'invalid {filter_name}', 'status': 'error'}), 400
    current_user_data = get_current_user()
    if not current_user_data:
        return jsonify({'error': '请先登录', 'status': 'unauthorized'}), 401

    try:
        engine = DatabasePoolManager.get_engine(db_name)
        where_clauses = [
            'db_name = :db_name',
            "search_mode = 'upload_match'",
        ]
        params = {'db_name': db_name, 'limit': limit}
        if not user_can_view_all(current_user_data):
            where_clauses.append('admin_id = :current_admin_id')
            params['current_admin_id'] = current_user_data.get('admin_id')
        if start_filter:
            where_clauses.append('created_at >= :start_time')
            params['start_time'] = start_filter
        if end_filter:
            end_dt = datetime.strptime(end_filter, '%Y-%m-%d') + timedelta(days=1)
            where_clauses.append('created_at < :end_time_plus')
            params['end_time_plus'] = end_dt.strftime('%Y-%m-%d')
        if workflow_mode_filter == 'ai':
            where_clauses.append(
                "(matched_tables LIKE :workflow_mode_ai_spaced OR matched_tables LIKE :workflow_mode_ai_compact)"
            )
            params['workflow_mode_ai_spaced'] = '%"workflow_mode": "ai"%'
            params['workflow_mode_ai_compact'] = '%"workflow_mode":"ai"%'
        elif workflow_mode_filter == 'template':
            where_clauses.append(
                "("
                "matched_tables LIKE :workflow_mode_template_spaced "
                "OR matched_tables LIKE :workflow_mode_template_compact "
                "OR (COALESCE(matched_tables, '') NOT LIKE :workflow_mode_ai_spaced "
                "AND COALESCE(matched_tables, '') NOT LIKE :workflow_mode_ai_compact)"
                ")"
            )
            params['workflow_mode_template_spaced'] = '%"workflow_mode": "template"%'
            params['workflow_mode_template_compact'] = '%"workflow_mode":"template"%'
            params['workflow_mode_ai_spaced'] = '%"workflow_mode": "ai"%'
            params['workflow_mode_ai_compact'] = '%"workflow_mode":"ai"%'
        where_sql = ' AND '.join(where_clauses)

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT nl_query, selected_table, matched_tables, generated_sql,
                       result_rows, execute_status, total_duration_ms, created_at,
                       admin_id, request_id, session_id, client_ip, user_agent,
                       action_type, visibility_scope
                FROM knowledge.query_log
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT :limit
            """), params).fetchall()

        history = []
        for row in rows:
            mapping = row._mapping
            payload = _parse_match_history_payload(mapping.get('matched_tables'))
            workflow_mode = str(payload.get('workflow_mode') or 'template').strip().lower()
            if workflow_mode not in {'template', 'ai'}:
                workflow_mode = 'template'
            if workflow_mode_filter and workflow_mode != workflow_mode_filter:
                continue
            matched_rows = payload.get('matched_rows')
            if matched_rows is None:
                matched_rows = mapping.get('result_rows')
            result_rows = payload.get('result_rows')
            if result_rows is None:
                result_rows = mapping.get('result_rows')
            source_table_name = payload.get('source_table_name') or mapping.get('selected_table') or ''
            template_label = payload.get('template_label') or payload.get('template_key') or ''
            matched_table_name = payload.get('matched_table_name') or ''
            matched_table_token = str(matched_table_name or '').strip()
            if matched_table_token.lower().startswith('tmp.'):
                matched_table_token = matched_table_token.split('.', 1)[1]
            result_sql = (
                _build_match_result_sql(matched_table_token)
                if matched_table_token
                else (payload.get('result_sql') or mapping.get('generated_sql') or '')
            )
            history.append({
                'title': mapping.get('nl_query') or f'{source_table_name} - {template_label}',
                'source_table_name': source_table_name,
                'source_table_label': payload.get('source_table_label') or source_table_name,
                'template_key': payload.get('template_key') or '',
                'template_label': template_label,
                'target_table_name': payload.get('target_table_name') or '',
                'matched_table_name': matched_table_name,
                'matched_rows': matched_rows,
                'result_rows': result_rows,
                'source_row_count': payload.get('source_row_count'),
                'keyword_column': payload.get('keyword_column') or '',
                'match_field': payload.get('match_field') or '',
                'field_mappings': payload.get('field_mappings') or [],
                'workflow_mode': workflow_mode,
                'result_sql': result_sql,
                'execute_status': mapping.get('execute_status') or '',
                'total_duration_ms': mapping.get('total_duration_ms'),
                'created_at': str(mapping.get('created_at')) if mapping.get('created_at') else None,
                'admin_id': mapping.get('admin_id'),
                'request_id': mapping.get('request_id'),
                'session_id': mapping.get('session_id'),
                'client_ip': mapping.get('client_ip'),
                'user_agent': mapping.get('user_agent'),
                'action_type': mapping.get('action_type'),
                'visibility_scope': mapping.get('visibility_scope'),
            })
        return jsonify({
            'status': 'success',
            'history': history,
            'filters': {
                'start_time': start_filter,
                'end_time': end_filter,
                'workflow_mode': workflow_mode_filter,
            },
        })
    except Exception as e:
        print(f"   ⚠️ 加载匹配历史失败（忽略，不影响主流程）: {e}")
        return jsonify({'status': 'success', 'history': []})


@main_bp.route('/api/upload_match_templates', methods=['GET'])
def upload_match_templates():
    """获取当前数据库下可用的上传匹配模板"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    try:
        db_config = load_upload_match_templates_config(db_name)
        include_disabled = request.args.get('include_disabled', '').strip().lower() in {'1', 'true', 'yes'}
        templates = get_upload_match_templates(db_name, db_config=db_config, enabled_only=not include_disabled)
        return jsonify({'status': 'success', 'templates': templates})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/upload_table_columns', methods=['GET'])
def upload_table_columns():
    """按需获取某张目标表的字段"""
    db_name = request.args.get('db_name', '').strip()
    table_name = request.args.get('table_name', '').strip()
    if not db_name or not table_name:
        return jsonify({'error': 'missing db_name or table_name', 'status': 'error'}), 400

    try:
        table_meta = _get_live_table_meta(db_name, table_name, default_schema='tmp')
        if not table_meta:
            tables = KnowledgeBase(db_name).get_table_list()
            table_meta = _resolve_table_meta(tables, table_name)
        if not table_meta:
            return jsonify({'error': f'未找到表: {table_name}', 'status': 'error'}), 404
        columns = table_meta.get('columns') or []
        return jsonify({
            'status': 'success',
            'table_name': table_name,
            'columns': columns,
            'suggested_keyword_column': _guess_keyword_column_from_columns(columns, table_name),
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/upload_excel_preview', methods=['POST'])
def upload_excel_preview():
    """预览上传 Excel 的字段"""
    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify({'error': '请选择文件', 'status': 'error'}), 400
    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        return jsonify({'error': '仅支持 .xlsx 或 .xls 格式', 'status': 'error'}), 400

    try:
        preview_df = pd.read_excel(file, nrows=5)
        preview_df.columns = [str(c).strip().replace(' ', '_') for c in preview_df.columns]
        columns = list(preview_df.columns)
        suggested_keyword_column = _guess_keyword_column(preview_df, '') if columns else None
        return jsonify({
            'status': 'success',
            'columns': columns,
            'suggested_keyword_column': suggested_keyword_column,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/insert_excel', methods=['POST'])
def insert_excel():
    """将 Excel 数据追加插入已有数据库表。"""
    db_name = request.form.get('db_name', '').strip()
    table_name = request.form.get('table_name', '').strip()
    file = request.files.get('file')
    if not db_name:
        return jsonify({'error': '请选择数据库', 'status': 'error'}), 400

    staging_table_name = ''
    insert_session_id = ''
    try:
        target_meta, df, column_mappings = _prepare_insert_excel_data(
            db_name,
            table_name,
            file,
        )
        full_table_name = _format_full_table_name(target_meta)
        staging_table_name = _create_insert_staging_table(db_name, target_meta, df)
        duplicate_summary = _analyze_insert_duplicates(
            db_name,
            target_meta,
            df,
            staging_table_name=staging_table_name,
        )
        duplicate_rows = duplicate_summary['total_rows']
        if duplicate_rows:
            insert_session_id = _store_insert_excel_session(
                db_name,
                target_meta,
                df,
                column_mappings,
                duplicate_summary,
                staging_table_name,
            )
        page, page_size = _parse_insert_duplicate_page_options(request.form)
        duplicate_payload = _build_insert_duplicate_response_payload(
            db_name,
            target_meta,
            df,
            column_mappings,
            duplicate_summary,
            page,
            page_size,
            mode='insert',
            insert_session_id=insert_session_id,
        )
        insertable_count = len(df) - duplicate_rows
        if insertable_count == 0:
            message = (
                f'发现 {duplicate_rows} 行重复 ID，本次没有新增行；'
                f'历史表重复 {duplicate_summary["existing_rows"]} 行，'
                f'Excel 内重复 {duplicate_summary["file_rows"]} 行。'
                '请在下方核对重复数据，勾选后确认更新'
            )
            if duplicate_rows > INSERT_RESPONSE_ROW_LIMIT:
                message += '；重复数据可分页查看'
            return jsonify({
                'status': 'success',
                **duplicate_payload,
                'table_name': full_table_name,
                'inserted_rows': 0,
                'message': message,
            })

        _insert_non_duplicate_rows_from_staging(
            db_name,
            target_meta,
            df,
            staging_table_name,
        )

        inserted_rows = insertable_count
        failed_hint = (
            '可下载目标表已有数据'
            if duplicate_summary['existing_rows']
            else '可下载重复数据明细'
        )
        message = (
            f'✅ 成功向 {full_table_name} 插入 {inserted_rows} 行，'
            f'{duplicate_rows} 行重复 ID 未插入，{failed_hint}'
            if duplicate_rows
            else f'✅ 成功向 {full_table_name} 插入 {inserted_rows} 行数据'
        )
        if duplicate_rows > INSERT_RESPONSE_ROW_LIMIT:
            message += '；重复数据可分页查看'
        return jsonify({
            'status': 'success',
            **duplicate_payload,
            'table_name': full_table_name,
            'inserted_rows': inserted_rows,
            'message': message,
        })
    except Exception as e:
        return jsonify({'error': _format_match_error(e), 'status': 'error'}), 400
    finally:
        if staging_table_name and not insert_session_id:
            _drop_insert_staging_table(
                db_name,
                target_meta if 'target_meta' in locals() else {},
                staging_table_name,
            )


@main_bp.route('/api/insert_excel_duplicates_page', methods=['POST'])
def insert_excel_duplicates_page():
    """按页返回插入流程中的重复数据核对明细。"""
    db_name = request.form.get('db_name', '').strip()
    table_name = request.form.get('table_name', '').strip()
    session_id = request.form.get('insert_session_id', '').strip()
    if not db_name:
        return jsonify({'error': '请选择数据库', 'status': 'error'}), 400

    try:
        page, page_size = _parse_insert_duplicate_page_options(request.form)
        session_item = _get_insert_excel_session(session_id, db_name, table_name)
        if not session_item:
            raise ValueError('重复数据会话不存在，请重新上传 Excel 后再查看')
        target_meta = session_item['target_meta']
        df = session_item['df'].copy()
        column_mappings = list(session_item.get('column_mappings') or [])
        duplicate_summary = session_item['duplicate_summary']

        duplicate_payload = _build_insert_duplicate_response_payload(
            db_name,
            target_meta,
            df,
            column_mappings,
            duplicate_summary,
            page,
            page_size,
            mode='duplicates_page',
            insert_session_id=session_id,
        )
        return jsonify({
            'status': 'success',
            **duplicate_payload,
        })
    except Exception as e:
        return jsonify({'error': _format_match_error(e), 'status': 'error'}), 400


@main_bp.route('/api/insert_excel_update_duplicates', methods=['POST'])
def insert_excel_update_duplicates():
    """人工确认后，用 Excel 中选择的重复行更新目标表已有数据。"""
    db_name = request.form.get('db_name', '').strip()
    table_name = request.form.get('table_name', '').strip()
    session_id = request.form.get('insert_session_id', '').strip()
    selected_rows_raw = request.form.get('selected_rows', '[]')
    select_all = str(request.form.get('select_all', '')).strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }
    if not db_name:
        return jsonify({'error': '请选择数据库', 'status': 'error'}), 400

    try:
        selected_rows = json.loads(selected_rows_raw or '[]')
        if not isinstance(selected_rows, list):
            raise ValueError('selected_rows must be a list')
    except Exception:
        return jsonify({'error': '更新行参数格式错误', 'status': 'error'}), 400

    try:
        session_item = _get_insert_excel_session(session_id, db_name, table_name)
        if not session_item:
            raise ValueError('重复数据会话不存在，请重新上传 Excel 后再更新')
        target_meta = session_item['target_meta']
        df = session_item['df'].copy()
        duplicate_summary = session_item['duplicate_summary']
        full_table_name = _format_full_table_name(target_meta)
        if select_all:
            selected_rows = list(
                duplicate_summary.get('existing_row_indices') or []
            )
            updated_rows = _update_insert_duplicate_rows_from_staging(
                db_name,
                target_meta,
                df,
                duplicate_summary,
            )
        else:
            updated_rows = _update_insert_duplicate_rows_from_staging(
                db_name,
                target_meta,
                df,
                duplicate_summary,
                selected_rows,
            )
        _remove_insert_duplicate_indices(duplicate_summary, selected_rows)
        if not duplicate_summary.get('total_rows'):
            _delete_insert_excel_session(session_id)
        return jsonify({
            'status': 'success',
            'mode': 'update_duplicates',
            'table_name': full_table_name,
            'updated_rows': updated_rows,
            'select_all': select_all,
            'insert_update_eligible_total': int(
                duplicate_summary.get('existing_rows') or 0
            ),
            'duplicate_summary': _public_insert_duplicate_summary(duplicate_summary),
            'message': f'✅ 已更新 {full_table_name} 中 {updated_rows} 行重复数据',
        })
    except Exception as e:
        return jsonify({'error': _format_match_error(e), 'status': 'error'}), 400


@main_bp.route('/api/upload_match_configs', methods=['GET', 'POST', 'PATCH', 'DELETE'])
def upload_match_configs_api():
    """读取、保存或删除上传匹配模板配置"""
    if request.method == 'GET':
        db_name = request.args.get('db_name', '').strip()
        if not db_name:
            return jsonify({'error': 'missing db_name', 'status': 'error'}), 400
        try:
            db_config = load_upload_match_templates_config(db_name)
            return jsonify({
                'status': 'success',
                'db_name': db_name,
                'db_config': db_config,
                'templates': get_upload_match_templates(db_name, db_config=db_config),
            })
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    data = request.get_json(silent=True) or {}
    db_name = (data.get('db_name') or '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name', 'status': 'error'}), 400

    try:
        db_config = dict(load_upload_match_templates_config(db_name))
        if request.method == 'DELETE':
            template_key = (data.get('template_key') or '').strip()
            if not is_current_upload_template_key(template_key):
                return jsonify({'error': 'missing template_key', 'status': 'error'}), 400
            delete_upload_match_template_record(db_name, template_key, current_user=get_current_user())
            db_config.pop(template_key, None)
            return jsonify({
                'status': 'success',
                'db_name': db_name,
                'db_config': db_config,
            })

        if request.method == 'PATCH':
            template_key = (data.get('template_key') or '').strip()
            if not is_current_upload_template_key(template_key):
                return jsonify({'error': 'missing template_key', 'status': 'error'}), 400
            if 'is_enabled' not in data:
                return jsonify({'error': 'missing is_enabled', 'status': 'error'}), 400
            if template_key not in db_config:
                return jsonify({'error': 'template not found', 'status': 'error'}), 404
            update_upload_match_template_enabled(
                db_name,
                template_key,
                data.get('is_enabled'),
                current_user=get_current_user(),
            )
            db_config = dict(load_upload_match_templates_config(db_name))
            return jsonify({
                'status': 'success',
                'db_name': db_name,
                'template_key': template_key,
                'is_enabled': bool((db_config.get(template_key) or {}).get('is_enabled', True)),
                'db_config': db_config,
                'templates': get_upload_match_templates(db_name, db_config=db_config),
            })

        template_key = (data.get('template_key') or '').strip()
        if not is_current_upload_template_key(template_key):
            return jsonify({'error': 'missing template_key', 'status': 'error'}), 400
        config = data.get('config')
        if not isinstance(config, dict):
            return jsonify({'error': 'missing config', 'status': 'error'}), 400

        existing_config = dict(db_config.get(template_key) or {})
        actor = _current_upload_template_actor(data)
        config = _merge_upload_template_config(template_key, dict(config), existing_config, actor=actor)
        duplicate_keys = _detect_duplicate_upload_template_label(db_config, template_key, config.get('label', ''))

        config['target_filter'] = _validate_target_filter(config.get('target_filter', ''))
        if not str(config.get('match_table') or '').strip():
            return jsonify({
                'status': 'error',
                'error': '目标表为空，请先选择或填写目标表',
            }), 400
        if not config.get('return_fields'):
            return jsonify({
                'status': 'error',
                'error': '字段映射为空，请先解析并填充字段后再保存',
            }), 400

        upsert_upload_match_template(db_name, template_key, config, current_user=get_current_user())
        db_config = dict(load_upload_match_templates_config(db_name))
        return jsonify({
            'status': 'success',
            'db_name': db_name,
            'template_key': template_key,
            'db_config': db_config,
            'templates': get_upload_match_templates(db_name, db_config=db_config),
            'duplicate_keys': duplicate_keys,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/match_uploaded_table', methods=['POST'])
def match_uploaded_table():
    """对已上传的历史表执行匹配"""
    start_time = time.time()
    data = request.get_json(silent=True) or {}
    request_id = (data.get('request_id') or '').strip() if isinstance(data.get('request_id'), str) else ''
    if not request_id:
        request_id = uuid.uuid4().hex
    db_name = (data.get('db_name') or '').strip()
    source_table_name = (data.get('source_table_name') or '').strip()
    source_table_label = str(data.get('source_table_label') or source_table_name).strip()
    template_key = (data.get('template_key') or '').strip()
    template_label = str(data.get('template_label') or template_key).strip()
    match_hint = (data.get('match_hint') or '').strip()
    workflow_mode = (data.get('workflow_mode', 'template') or 'template').strip().lower()
    if workflow_mode not in {'template', 'ai'}:
        workflow_mode = 'template'
    use_template_mode = workflow_mode != 'ai'
    ai_mode = workflow_mode == 'ai'
    keyword_column = (data.get('keyword_column') or '').strip().replace(' ', '_')
    match_field_input = (data.get('match_field') or '').strip()
    field_mappings = _normalize_match_field_mappings(
        data.get('field_mappings'),
        keyword_column,
        match_field_input,
    )
    return_fields_raw = (data.get('return_fields') or '').strip()
    match_mode = (data.get('match_mode', 'exact') or 'exact').strip().lower()
    schema_filter = (data.get('schema_filter') or '').strip() or None
    auto_match_table = str(data.get('auto_match_table', '1')).strip().lower() not in {'0', 'false', 'no', 'off'}

    if not db_name:
        return jsonify({'error': '请选择数据库', 'status': 'error'}), 400
    if not source_table_name:
        return jsonify({'error': '请选择目标表', 'status': 'error'}), 400

    try:
        tables = KnowledgeBase(db_name).get_table_list()
        source_table_meta = _get_live_table_meta(db_name, source_table_name, default_schema='tmp')
        if not source_table_meta:
            source_table_meta = _resolve_table_meta(tables, source_table_name)
        if not source_table_meta:
            return jsonify({'error': f'未找到目标表: {source_table_name}', 'status': 'error'}), 404

        source_columns = source_table_meta.get('columns') or []
        configured_upload_match_config = get_upload_match_config_from_db(
            db_name,
            template_key=template_key if use_template_mode else None,
        )
        if not template_label:
            template_label = (
                str(configured_upload_match_config.get('label') or '').strip()
                or str(configured_upload_match_config.get('name') or '').strip()
                or template_key
            )

        configured_keyword_column = str(configured_upload_match_config.get('keyword_column', '') or '').strip().replace(' ', '_')
        if field_mappings and field_mappings[0].get('source_field') and not keyword_column:
            keyword_column = field_mappings[0]['source_field']
        if not keyword_column:
            if use_template_mode and configured_keyword_column:
                keyword_column = configured_keyword_column
            else:
                keyword_column = _guess_keyword_column_from_columns(source_columns, match_hint or source_table_name)

        if not keyword_column:
            raise ValueError('未能识别目标表字段，请先选择字段，或在模板中配置目标表字段')
        if keyword_column not in source_columns:
            return jsonify({'error': f'目标表字段 {keyword_column} 不在目标表字段中', 'status': 'error'}), 400

        if not field_mappings:
            field_mappings = _normalize_match_field_mappings(None, keyword_column, match_field_input)
        else:
            if not field_mappings[0].get('source_field'):
                field_mappings[0]['source_field'] = keyword_column
            for index, mapping in enumerate(field_mappings):
                source_field = mapping.get('source_field') or ''
                if not source_field:
                    raise ValueError(f'第 {index + 1} 组匹配缺少目标表字段')
                if source_field not in source_columns:
                    return jsonify({'error': f'目标表字段 {source_field} 不在目标表字段中', 'status': 'error'}), 400

        match_plan = _resolve_upload_match_plan(
            db_name,
            source_table_name,
            source_columns,
            keyword_column,
            match_hint or source_table_name,
            template_key=template_key,
            match_field_input=match_field_input,
            field_mappings=field_mappings,
            return_fields_raw=return_fields_raw,
            match_mode=match_mode,
            schema_filter=schema_filter,
            auto_match_table=auto_match_table,
            use_template_mode=use_template_mode,
            ai_mode=ai_mode,
            match_config=configured_upload_match_config,
        )

        engine = DatabasePoolManager.get_engine(db_name)
        source_row_count = None
        with engine.connect() as conn:
            try:
                source_row_count = conn.execute(text(f'SELECT COUNT(*) FROM tmp.{_quote_ident(source_table_name)}')).fetchone()[0]
            except Exception:
                source_row_count = None

        matched_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        source_base_name = _extract_upload_base_name(source_table_name) or source_table_name
        matched_table_name = _build_timestamped_table_name(source_base_name, matched_timestamp, '_matched')
        match_result = _execute_match_query(
            db_name,
            source_table_name,
            matched_table_name,
            keyword_column,
            match_plan['target_table_meta'],
            match_plan['match_field_spec'],
            match_plan['return_field_specs'],
            match_plan['match_mode'],
            match_plan['configured_target_filter'] if use_template_mode else '',
            source_columns,
            match_plan['match_field_pairs'],
            match_plan.get('target_sql_text', '') if use_template_mode else '',
        )
        total_duration_ms = (time.time() - start_time) * 1000
        matched_table_full_name = f'tmp.{matched_table_name}'
        matched_rows = int(match_result['matched_rows'] or 0)
        source_rows = int(source_row_count or 0) if source_row_count is not None else None
        result_sql = match_result['result_query']
        history_template_label = template_label or template_key or ('AI 智能匹配' if ai_mode else '未选择业务模板')
        history_payload = {
            'source_table_name': source_table_name,
            'source_table_label': source_table_label,
            'template_key': template_key,
            'template_label': history_template_label,
            'target_table_name': _format_full_table_name(match_plan['target_table_meta']),
            'matched_table_name': matched_table_full_name,
            'matched_rows': matched_rows,
            'result_rows': int(match_result['result_rows'] or 0),
            'source_row_count': source_rows,
            'keyword_column': keyword_column,
            'match_field': match_plan['match_field'],
            'field_mappings': match_plan['field_mappings'],
            'workflow_mode': workflow_mode,
            'result_sql': result_sql,
        }
        insert_query_log(db_name, {
            'db_name': db_name,
            'nl_query': f'{source_table_label} - {history_template_label}',
            'schema_filter': schema_filter,
            'search_mode': 'upload_match',
            'selected_table': source_table_name,
            'top_k': None,
            'matched_tables': history_payload,
            'generated_sql': result_sql,
            'execute_status': 'success',
            'error_message': None,
            'result_rows': int(match_result['result_rows'] or 0),
            'search_duration_ms': 0,
            'llm_duration_ms': 0,
            'sql_exec_duration_ms': total_duration_ms,
            'total_duration_ms': total_duration_ms,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'llm_calls': 0,
            'request_id': request_id,
            'action_type': 'upload_match',
        })

        return jsonify({
            'status': 'success',
            'request_id': request_id,
            'mode': 'match',
            'workflow_mode': workflow_mode,
            'source_table_name': source_table_name,
            'match_status': 'success',
            'match_mode': match_plan['match_mode'],
            'keyword_column': keyword_column,
            'match_field': match_plan['match_field'],
            'field_mappings': match_plan['field_mappings'],
            'target_table_name': _format_full_table_name(match_plan['target_table_meta']),
            'matched_table_name': matched_table_full_name,
            'matched_rows': matched_rows,
            'result_rows': int(match_result['result_rows'] or 0),
            'source_row_count': source_rows,
            'preview_columns': list(match_result['preview_df'].columns),
            'preview_rows': safe_records(match_result['preview_df']),
            'generated_sql': match_result['match_query'].strip(),
            'result_sql': result_sql,
            'message': f'✅ 已对 {source_table_name} 完成匹配，命中 {matched_rows} 行结果'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/api/upload_table_history', methods=['GET'])
def upload_table_history():
    """获取当前数据库下历史上传表名"""
    db_name = request.args.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400

    try:
        limit = int(request.args.get('limit', 20) or 20)
    except Exception:
        limit = 20
    limit = max(1, min(limit, 100))

    try:
        history = _get_live_upload_tables(db_name, limit)
        if not history:
            history = _get_cached_upload_tables(db_name, limit)
        return jsonify({'status': 'success', 'tables': history})
    except Exception as e:
        print(f"❌ 加载上传历史失败: {e}")
        try:
            return jsonify({'status': 'success', 'tables': _get_cached_upload_tables(db_name, limit)})
        except Exception:
            return jsonify({'status': 'success', 'tables': []})


@main_bp.route('/execute_nl_query', methods=['POST'])
def handle_nl_query():
    """处理自然语言查询"""
    data = request.get_json() or {}
    # 优先使用前端传的 request_id（用于取消），否则后端生成
    client_request_id = (data.get('request_id') or '').strip() if isinstance(data.get('request_id'), str) else ''
    if client_request_id:
        token = cancel_registry.register_with_id(client_request_id)
        request_id, cancel_token = client_request_id, token
    else:
        request_id, cancel_token = cancel_registry.create()

    try:
        # 限流检查
        if not _nl_query_limiter.allow():
            return jsonify({'error': '请求过于频繁，请稍后再试', 'status': 'error'}), 429

        if 'nl_query' not in data:
            return jsonify({'error': 'Missing nl_query'}), 400

        db_name = data.get('db_name')
        if not db_name:
            return jsonify({'error': 'Missing db_name'}), 400

        selected_table = data.get('selected_table')
        if selected_table and isinstance(selected_table, str):
            selected_table = selected_table.strip() or None
        else:
            selected_table = None

        schema_name = data.get('schema_name', '').strip() or None
        use_vector_search = data.get('use_vector_search', True)
        top_k_tables = data.get('top_k_tables', 10)
        embedding_provider = data.get('embedding_provider', '').strip() or None
        query_context_mode = (
            data.get('query_context_mode')
            or data.get('context_mode')
            or data.get('mode')
            or ''
        )
        query_context_mode = str(query_context_mode).strip() or None

        print(f"\n{'='*60}")
        print(f"📨 查询: {db_name} - {data['nl_query'][:100]}...")
        print(f"   request_id: {request_id}")
        print(f"   向量检索: {use_vector_search}, 指定表: {selected_table}, schema: {schema_name}")
        print(f"   向量模型: {embedding_provider or '默认'}")
        print(f"   上下文模式: {query_context_mode or schema_name or '默认'}")
        print(f"{'='*60}")

        current_user_data = get_current_user()
        converter = TextToSQLConverter(db_name, current_user=current_user_data)

        start_time = time.time()
        sql, result = converter.execute_nl_query(
            nl_query=data['nl_query'],
            selected_table=selected_table,
            use_vector_search=use_vector_search,
            top_k_tables=top_k_tables,
            schema_filter=schema_name,
            cancel_token=cancel_token,
            embedding_provider=embedding_provider,
            request_id=request_id,
            query_context_mode=query_context_mode,
        )

        elapsed = (time.time() - start_time) * 1000
        print(f"✅ 查询完成，总耗时: {elapsed:.2f} ms")

        MAX_DISPLAY_ROWS = 500
        total_rows = len(result)
        if total_rows > MAX_DISPLAY_ROWS:
            display_result = safe_records(result.head(MAX_DISPLAY_ROWS))
        else:
            display_result = safe_records(result)

        return jsonify({
            'request_id': request_id,
            'generated_sql': sql,
            'sql_result': display_result,
            'columns': list(result.columns),
            'total_rows': total_rows,
            'status': 'success'
        })
    except CancelledError as e:
        print(f"⛔ 查询被用户取消: {request_id}")
        return jsonify({'error': '查询已取消', 'status': 'cancelled', 'request_id': request_id}), 499
    except Exception as e:
        print(f"❌ 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error', 'request_id': request_id}), 500
    finally:
        cancel_registry.cleanup(request_id)


@main_bp.route('/api/rebuild_all_vectors', methods=['POST'])
def rebuild_all_vectors():
    """手动触发全量向量重建"""
    data = request.get_json() or {}
    db_name = data.get('db_name', '').strip()
    if not db_name:
        return jsonify({'error': 'missing db_name'}), 400
    try:
        from core.knowledge import KnowledgeBase
        kb = KnowledgeBase(db_name)
        table_records, vector_texts = kb.get_vector_texts()
        if not table_records:
            return jsonify({'status': 'success', 'message': '无表需更新'})
        for provider, model in iter_embedding_models():
            kb.save_embeddings_incrementally(model, table_records, vector_texts)
        kb.rebuild_knowledge_vectors()
        return jsonify({'status': 'success', 'message': f'向量更新完成 ({db_name})'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@main_bp.route('/cancel_query', methods=['POST'])
def cancel_query():
    """取消正在执行的查询"""
    data = request.get_json() or {}
    request_id = data.get('request_id')
    if not request_id:
        return jsonify({'error': 'Missing request_id', 'status': 'error'}), 400
    found = cancel_registry.cancel(request_id)
    if found:
        print(f"⛔ 收到取消请求: {request_id}")
        return jsonify({'status': 'cancelled', 'request_id': request_id})
    return jsonify({'status': 'not_found', 'request_id': request_id}), 404


@main_bp.route('/api/upload_excel', methods=['POST'])
def upload_excel():
    """上传 Excel，并可按关键词自动匹配其他表字段"""
    db_name = request.form.get('db_name', '').strip()
    table_name = request.form.get('table_name', '').strip()
    file = request.files.get('file')
    template_key = request.form.get('template_key', '').strip()
    match_hint = request.form.get('match_hint', '').strip()
    workflow_mode = (request.form.get('workflow_mode', 'template') or 'template').strip().lower()
    if workflow_mode not in {'template', 'ai'}:
        workflow_mode = 'template'
    use_template_mode = workflow_mode != 'ai'
    ai_mode = workflow_mode == 'ai'
    keyword_column = request.form.get('keyword_column', '').strip().replace(' ', '_')
    match_table_input = request.form.get('match_table', '').strip()
    match_field_input = request.form.get('match_field', '').strip()
    field_mappings = _normalize_match_field_mappings(
        request.form.get('field_mappings'),
        keyword_column,
        match_field_input,
    )
    return_fields_raw = request.form.get('return_fields', '').strip()
    match_mode = (request.form.get('match_mode', 'exact') or 'exact').strip().lower()
    schema_filter = request.form.get('schema_filter', '').strip() or None
    auto_match_table = (request.form.get('auto_match_table', '1') or '1').strip().lower() not in {'0', 'false', 'no', 'off'}
    do_match = (request.form.get('do_match', '1') or '1').strip().lower() not in {'0', 'false', 'no', 'off'}

    if not db_name:
        return jsonify({'error': '请选择数据库', 'status': 'error'}), 400
    if not file or not file.filename:
        return jsonify({'error': '请选择文件', 'status': 'error'}), 400
    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        return jsonify({'error': '仅支持 .xlsx 或 .xls 格式', 'status': 'error'}), 400

    upload_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    effective_table_name = table_name or os.path.splitext(os.path.basename(file.filename))[0]
    effective_table_name = effective_table_name.strip() or 'upload'
    config_template_key = template_key if use_template_mode else None
    upload_match_config = get_upload_match_config_from_db(
        db_name,
        template_key=config_template_key,
    )

    safe_name = _build_timestamped_table_name(effective_table_name, upload_timestamp)
    if not safe_name:
        return jsonify({'error': '表名不合法，请使用字母、数字、下划线', 'status': 'error'}), 400

    try:
        df = pd.read_excel(file)
        if df.empty:
            return jsonify({'error': 'Excel 文件为空', 'status': 'error'}), 400

        df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]

        if do_match:
            configured_keyword_column = str(upload_match_config.get('keyword_column', '') or '').strip().replace(' ', '_')
            if field_mappings and field_mappings[0].get('source_field') and not keyword_column:
                keyword_column = field_mappings[0]['source_field']
            if not keyword_column:
                if use_template_mode and configured_keyword_column:
                    keyword_column = configured_keyword_column
                else:
                    keyword_column = _guess_keyword_column(df, match_hint or effective_table_name)

            if not keyword_column:
                raise ValueError('未能识别目标表字段，请在下拉中选择，或在 AI 说明中补充提示')
            if keyword_column not in df.columns:
                return jsonify({'error': f'目标表字段 {keyword_column} 不在 Excel 表头中', 'status': 'error'}), 400

            if not field_mappings:
                field_mappings = _normalize_match_field_mappings(None, keyword_column, match_field_input)
            else:
                if not field_mappings[0].get('source_field'):
                    field_mappings[0]['source_field'] = keyword_column
                for index, mapping in enumerate(field_mappings):
                    source_field = mapping.get('source_field') or ''
                    if not source_field:
                        raise ValueError(f'第 {index + 1} 组匹配缺少目标表字段')
                    if source_field not in df.columns:
                        return jsonify({'error': f'目标表字段 {source_field} 不在 Excel 表头中', 'status': 'error'}), 400

        engine = DatabasePoolManager.get_engine(db_name)

        # Phase 1: DDL - DROP + CREATE 在独立事务中
        type_map = {
            'int64': 'BIGINT', 'Int64': 'BIGINT',
            'float64': 'DOUBLE PRECISION', 'Float64': 'DOUBLE PRECISION',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'object': 'TEXT',
        }
        cols = []
        for name, dtype in df.dtypes.items():
            pg_type = type_map.get(str(dtype), 'TEXT')
            safe_col = str(name).replace('"', '""')
            cols.append(f'"{safe_col}" {pg_type}')
        with engine.connect() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS tmp."{safe_name}" CASCADE'))
            conn.execute(text(f'CREATE TABLE tmp."{safe_name}" ({", ".join(cols)})'))
            conn.commit()

        # Phase 2: DML - INSERT 在独立事务中
        df.to_sql(safe_name, engine, schema='tmp', if_exists='append', index=False, method=None)

        row_count = len(df)
        full_name = f"tmp.{safe_name}"
        response_payload = {
            'status': 'success',
            'mode': 'upload',
            'workflow_mode': workflow_mode,
            'table_name': full_name,
            'row_count': row_count,
            'columns': list(df.columns),
            'message': f'✅ 成功导入 {row_count} 行数据到 {full_name}'
        }

        if not do_match:
            response_payload.update({
                'workflow_mode': workflow_mode,
                'match_status': 'skipped',
                'message': f'✅ 成功导入 {row_count} 行数据到 {full_name}'
            })
            return jsonify(response_payload)

        try:
            keyword_samples = _sample_keyword_values(df, keyword_column, limit=5)
            match_plan = _resolve_upload_match_plan(
                db_name,
                safe_name,
                list(df.columns),
                keyword_column,
                match_hint or effective_table_name,
                template_key=template_key,
                match_table_input=match_table_input,
                match_field_input=match_field_input,
                field_mappings=field_mappings,
                return_fields_raw=return_fields_raw,
                match_mode=match_mode,
                schema_filter=schema_filter,
                auto_match_table=auto_match_table,
                use_template_mode=use_template_mode,
                ai_mode=ai_mode,
                sample_keywords=keyword_samples,
                match_config=upload_match_config,
            )

            matched_table_name = _build_timestamped_table_name(effective_table_name, upload_timestamp, '_matched')
            match_result = _execute_match_query(
                db_name,
                safe_name,
                matched_table_name,
                keyword_column,
                match_plan['target_table_meta'],
                match_plan['match_field_spec'],
                match_plan['return_field_specs'],
                match_plan['match_mode'],
                match_plan['configured_target_filter'] if use_template_mode else '',
                list(df.columns),
                match_plan['match_field_pairs'],
                match_plan.get('target_sql_text', '') if use_template_mode else '',
            )

            response_payload.update({
                'mode': 'match',
                'workflow_mode': workflow_mode,
                'match_status': 'success',
                'match_mode': match_plan['match_mode'],
                'keyword_column': keyword_column,
                'match_field': match_plan['match_field'],
                'field_mappings': match_plan['field_mappings'],
                'target_table_name': _format_full_table_name(match_plan['target_table_meta']),
                'matched_rows': int(match_result['matched_rows'] or 0),
                'result_rows': int(match_result['result_rows'] or 0),
                'preview_columns': list(match_result['preview_df'].columns),
                'preview_rows': safe_records(match_result['preview_df']),
                'generated_sql': match_result['match_query'].strip(),
                'result_sql': match_result['result_query'],
                'message': (
                    f'✅ 成功导入 {row_count} 行数据到 {full_name}，'
                    f'并匹配到 {int(match_result["matched_rows"] or 0)} 行结果'
                )
            })
            return jsonify(response_payload)
        except Exception as match_error:
            match_error_message = _format_match_error(match_error)
            response_payload.update({
                'mode': 'upload',
                'workflow_mode': workflow_mode,
                'match_status': 'failed',
                'match_error': match_error_message,
                'message': f'✅ 成功导入 {row_count} 行数据到 {full_name}，但匹配失败：{match_error_message}'
            })
            return jsonify(response_payload)
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500
