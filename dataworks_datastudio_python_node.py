
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import ssl
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import psycopg2
from psycopg2.extras import Json, execute_values

DEFAULT_FILE_TYPES = '6,10,24,1093,221,1221,225,227,228,229,230,257,258,259,260'
DEFAULT_PAGE_SIZE = 100
DEFAULT_MAX_PAGES = 500
DEFAULT_CONTENT_LIMIT = 8000
DEFAULT_REGION_ID = 'cn-beijing'
DEFAULT_ENDPOINT = 'https://dataworks.{region}.aliyuncs.com'
DEFAULT_HTTP_TIMEOUT_SECONDS = 30
DEFAULT_HOLOGRES_PORT = '80'
DEFAULT_HOLOGRES_SSLMODE = 'prefer'
DEFAULT_HOLOGRES_SCHEMA = 'knowledge'
DEFAULT_HOLOGRES_TABLE = 'dataworks_node_knowledge'
DEFAULT_KNOWLEDGE_DB_NAME = 'hologres'
DEFAULT_DATAWORKS_MAX_RETRIES = 0
DEFAULT_DATAWORKS_RETRY_BASE_SECONDS = 3.0
DEFAULT_DATAWORKS_RETRY_MAX_SECONDS = 60.0
DEFAULT_DATAWORKS_REQUEST_INTERVAL_SECONDS = 0.2
DEFAULT_DATAWORKS_USE_TYPE = 'NORMAL'

MANUAL_CONFIG = {
    'DATAWORKS_ACCESS_KEY_ID': '',
    'DATAWORKS_ACCESS_KEY_SECRET': '',
    'DATAWORKS_PROJECT_ID': '333511',
    'HOLOGRES_HOST': 'your-hologres-host',
    'HOLOGRES_DB': 'your-hologres-db',
    'HOLOGRES_USER': 'your-hologres-user',
    'HOLOGRES_PASSWORD': '',
    'DATAWORKS_MAX_RETRIES': str(DEFAULT_DATAWORKS_MAX_RETRIES),
    'DATAWORKS_RETRY_BASE_SECONDS': str(DEFAULT_DATAWORKS_RETRY_BASE_SECONDS),
    'DATAWORKS_RETRY_MAX_SECONDS': str(DEFAULT_DATAWORKS_RETRY_MAX_SECONDS),
    'DATAWORKS_REQUEST_INTERVAL_SECONDS': str(DEFAULT_DATAWORKS_REQUEST_INTERVAL_SECONDS),
    'DATAWORKS_USE_TYPE': DEFAULT_DATAWORKS_USE_TYPE,
}


def _json_default(value: Any):
    if isinstance(value, datetime):
        return value.isoformat(sep=' ', timespec='seconds')
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _compact_text(text_value: Any, limit: int = DEFAULT_CONTENT_LIMIT) -> str:
    value = str(text_value or '').strip()
    if len(value) <= limit:
        return value
    return value[:limit] + '\n...(内容已截断)'


def _normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _normalize_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        text_value = value.strip()
        if not text_value:
            return []
        try:
            parsed = json.loads(text_value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [item.strip() for item in re.split(r'[,\n，；;]+', text_value) if item.strip()]
    return [value]


def _split_ids(raw: Any) -> List[str]:
    result = []
    for item in _normalize_list(raw):
        value = _normalize_text(item)
        if value:
            result.append(value)
    return list(dict.fromkeys(result))


def _safe_json_hash(value: Any) -> str:
    return hashlib.md5(_json_dumps(value).encode('utf-8')).hexdigest()


def _extract_mapping(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get('Data')
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get('File'), dict):
        merged = dict(data['File'])
        for key, value in data.items():
            if key != 'File' and key not in merged:
                merged[key] = value
        return merged
    return dict(data)


def _extract_content(payload_file: Dict[str, Any]) -> str:
    for key in ('Content', 'content', 'ScriptContent', 'scriptContent'):
        value = payload_file.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ''


def _extract_node_config(payload_file: Dict[str, Any]) -> Dict[str, Any]:
    for key in ('NodeConfiguration', 'nodeConfiguration', 'Spec', 'spec'):
        value = payload_file.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return {}


def _extract_table_names(entries: Any) -> List[str]:
    names: List[str] = []
    for item in _normalize_list(entries):
        if isinstance(item, dict):
            for key in (
                'RefTableName', 'refTableName', 'TableName', 'tableName',
                'Table', 'table', 'Name', 'name', 'OutputTableName', 'outputTableName',
                'Input', 'input', 'Output', 'output', 'Str', 'str', 'Value', 'value',
            ):
                value = _normalize_text(item.get(key))
                if value:
                    names.append(value)
                    break
            else:
                table_guid = _normalize_text(item.get('Guid') or item.get('guid'))
                if table_guid:
                    names.append(table_guid)
        else:
            value = _normalize_text(item)
            if value:
                names.append(value)
    return list(dict.fromkeys(names))


def _extract_file_list(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    data = payload.get('Data')
    if not isinstance(data, dict):
        return [], 0
    files = data.get('Files') or data.get('FileList') or data.get('FileInfos') or []
    if isinstance(files, dict):
        files = [files]
    if not isinstance(files, list):
        files = []
    total_count = data.get('TotalCount') or data.get('Count') or len(files)
    try:
        total_count = int(total_count)
    except Exception:
        total_count = len(files)
    return [item for item in files if isinstance(item, dict)], total_count


def _extract_dependency_page(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    paging = payload.get('PagingInfo')
    if not isinstance(paging, dict):
        data = payload.get('Data')
        if isinstance(data, dict):
            paging = data.get('PagingInfo') if isinstance(data.get('PagingInfo'), dict) else data
    if not isinstance(paging, dict):
        return [], 0
    nodes = paging.get('Nodes') or paging.get('DependentNodes') or []
    if isinstance(nodes, dict):
        nodes = [nodes]
    if not isinstance(nodes, list):
        nodes = []
    total_count = paging.get('TotalCount') or len(nodes)
    try:
        total_count = int(total_count)
    except Exception:
        total_count = len(nodes)
    return [item for item in nodes if isinstance(item, dict)], total_count


def _extract_table_refs(block: Any) -> List[str]:
    if not isinstance(block, dict):
        return []
    tables: List[str] = []
    for entry in _normalize_list(block.get('NodeOutputs') or []):
        if isinstance(entry, dict):
            ref_table = _normalize_text(entry.get('RefTableName') or entry.get('refTableName'))
            if ref_table:
                tables.append(ref_table)
    for entry in _normalize_list(block.get('Tables') or []):
        if isinstance(entry, dict):
            guid = _normalize_text(entry.get('Guid') or entry.get('guid'))
            if guid:
                tables.append(guid)
    return list(dict.fromkeys([table for table in tables if table]))


def _normalize_dependency_node(node: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    node_id = _normalize_text(node.get('Id') or node.get('id') or node.get('NodeId') or node.get('node_id'))
    node_name = _normalize_text(node.get('Name') or node.get('name') or node.get('NodeName') or node.get('node_name'))
    script = node.get('Script') if isinstance(node.get('Script'), dict) else {}
    absolute_folder_path = _normalize_text(
        node.get('AbsoluteFolderPath')
        or node.get('absolute_folder_path')
        or script.get('Path')
        or script.get('path')
        or node.get('Path')
        or node.get('path')
    )
    input_tables = _extract_table_refs(node.get('Inputs'))
    output_tables = _extract_table_refs(node.get('Outputs'))
    dependent_node_ids = _split_ids(
        node.get('DependentNodeIdList')
        or node.get('dependentNodeIdList')
        or node.get('DependentNodeIds')
        or node.get('dependentNodeIds')
    )
    return {
        'node_id': node_id,
        'node_key': node_id,
        'node_name': node_name,
        'file_name': node_name,
        'absolute_folder_path': absolute_folder_path,
        'input_tables': input_tables,
        'output_tables': output_tables,
        'dependent_node_ids': dependent_node_ids,
        'node_configuration': node,
    }


def _percent_encode(value: Any) -> str:
    return quote(str(value), safe='-_.~')


@dataclass
class RuntimeConfig:
    access_key_id: str = ''
    access_key_secret: str = ''
    project_id: Optional[int] = None
    hologres_host: str = ''
    hologres_db: str = ''
    hologres_user: str = ''
    hologres_password: str = ''
    dataworks_max_retries: int = DEFAULT_DATAWORKS_MAX_RETRIES
    dataworks_retry_base_seconds: float = DEFAULT_DATAWORKS_RETRY_BASE_SECONDS
    dataworks_retry_max_seconds: float = DEFAULT_DATAWORKS_RETRY_MAX_SECONDS
    dataworks_request_interval_seconds: float = DEFAULT_DATAWORKS_REQUEST_INTERVAL_SECONDS
    dataworks_use_type: str = DEFAULT_DATAWORKS_USE_TYPE
    dry_run: bool = False

    @classmethod
    def from_argv(cls, argv: Sequence[str]) -> 'RuntimeConfig':
        dry_run = False
        if '--dry-run' in argv:
            dry_run = True

        def pick(name: str, default: str = '') -> str:
            value = os.getenv(name)
            if value is None or not str(value).strip():
                value = MANUAL_CONFIG.get(name, default)
            if value is None:
                return default
            text = str(value).strip()
            return text if text else default

        def pick_int(name: str, default: int) -> int:
            try:
                return int(pick(name, str(default)))
            except Exception:
                return default

        def pick_float(name: str, default: float) -> float:
            try:
                return float(pick(name, str(default)))
            except Exception:
                return default

        project_id_raw = pick('DATAWORKS_PROJECT_ID', '')
        project_id = None
        if project_id_raw:
            try:
                project_id = int(project_id_raw)
            except Exception:
                project_id = None

        return cls(
            access_key_id=pick('DATAWORKS_ACCESS_KEY_ID'),
            access_key_secret=pick('DATAWORKS_ACCESS_KEY_SECRET'),
            project_id=project_id,
            hologres_host=pick('HOLOGRES_HOST'),
            hologres_db=pick('HOLOGRES_DB'),
            hologres_user=pick('HOLOGRES_USER'),
            hologres_password=pick('HOLOGRES_PASSWORD'),
            dataworks_max_retries=max(0, pick_int('DATAWORKS_MAX_RETRIES', DEFAULT_DATAWORKS_MAX_RETRIES)),
            dataworks_retry_base_seconds=max(
                0.1,
                pick_float('DATAWORKS_RETRY_BASE_SECONDS', DEFAULT_DATAWORKS_RETRY_BASE_SECONDS),
            ),
            dataworks_retry_max_seconds=max(
                1.0,
                pick_float('DATAWORKS_RETRY_MAX_SECONDS', DEFAULT_DATAWORKS_RETRY_MAX_SECONDS),
            ),
            dataworks_request_interval_seconds=max(
                0.0,
                pick_float('DATAWORKS_REQUEST_INTERVAL_SECONDS', DEFAULT_DATAWORKS_REQUEST_INTERVAL_SECONDS),
            ),
            dataworks_use_type=pick('DATAWORKS_USE_TYPE', DEFAULT_DATAWORKS_USE_TYPE),
            dry_run=dry_run,
        )

    def resolved_endpoint(self) -> str:
        return DEFAULT_ENDPOINT.format(region=DEFAULT_REGION_ID)


class DataWorksApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        action: str = '',
        status_code: Optional[int] = None,
        error_code: str = '',
        request_id: str = '',
        response_message: str = '',
    ):
        super().__init__(message)
        self.action = action
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id
        self.response_message = response_message

    @property
    def is_throttling(self) -> bool:
        text = f'{self.error_code} {self.response_message} {self}'.lower()
        return self.status_code == 429 or 'throttl' in text or 'rate limit' in text


class DataWorksOpenAPIClient:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.timeout = DEFAULT_HTTP_TIMEOUT_SECONDS

    def is_configured(self) -> bool:
        return bool(self.cfg.access_key_id and self.cfg.access_key_secret and self.cfg.project_id is not None)

    def _sign(self, params: Dict[str, Any], method: str) -> str:
        canonical = '&'.join(
            f"{_percent_encode(key)}={_percent_encode(value)}"
            for key, value in sorted(params.items())
            if value is not None and value != ''
        )
        string_to_sign = f"{method.upper()}&{_percent_encode('/')}&{_percent_encode(canonical)}"
        key = f"{self.cfg.access_key_secret}&".encode('utf-8')
        digest = hmac.new(key, string_to_sign.encode('utf-8'), hashlib.sha1).digest()
        return base64.b64encode(digest).decode('utf-8')

    def _build_api_error(
        self,
        action: str,
        *,
        status_code: Optional[int] = None,
        error_code: str = '',
        request_id: str = '',
        response_message: str = '',
    ) -> DataWorksApiError:
        parts = [action]
        if status_code is not None:
            parts.append(f'HTTP {status_code}')
        if error_code:
            parts.append(f'code={error_code}')
        if request_id:
            parts.append(f'request_id={request_id}')
        parts.append(f'message={response_message or "<empty>"}')
        return DataWorksApiError(
            ', '.join(parts),
            action=action,
            status_code=status_code,
            error_code=error_code,
            request_id=request_id,
            response_message=response_message,
        )

    def _request_once(self, action: str, params: Optional[Dict[str, Any]] = None, method: str = 'POST') -> Dict[str, Any]:
        if not self.is_configured():
            raise DataWorksApiError('DataWorks 凭证或项目配置不完整')

        payload: Dict[str, Any] = {
            'Action': action,
            'Version': '2024-05-18',
            'Format': 'JSON',
            'SignatureMethod': 'HMAC-SHA1',
            'SignatureVersion': '1.0',
            'SignatureNonce': uuid.uuid4().hex,
            'Timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'RegionId': DEFAULT_REGION_ID,
            'AccessKeyId': self.cfg.access_key_id,
            'ProjectId': self.cfg.project_id,
        }
        for key, value in (params or {}).items():
            if value is not None and value != '':
                payload[key] = value

        signature = self._sign(payload, method)
        signed_payload = dict(payload)
        signed_payload['Signature'] = signature
        body = urlencode(signed_payload).encode('utf-8')
        request = Request(self.cfg.resolved_endpoint(), data=body, method=method.upper())
        request.add_header('Content-Type', 'application/x-www-form-urlencoded')
        try:
            with urlopen(request, timeout=self.timeout, context=ssl.create_default_context()) as resp:
                raw = resp.read().decode('utf-8')
        except HTTPError as exc:
            raw = exc.read().decode('utf-8', errors='replace')
            message = raw[:500] if raw else exc.reason
            error_code = ''
            request_id = ''
            try:
                error_payload = json.loads(raw)
                if isinstance(error_payload, dict):
                    error_code = _normalize_text(error_payload.get('ErrorCode') or error_payload.get('Code'))
                    request_id = _normalize_text(error_payload.get('RequestId'))
                    message = (
                        error_payload.get('ErrorMessage')
                        or error_payload.get('Message')
                        or error_payload.get('ErrorCode')
                        or message
                    )
            except Exception:
                pass
            raise self._build_api_error(
                action,
                status_code=exc.code,
                error_code=error_code,
                request_id=request_id,
                response_message=_normalize_text(message),
            ) from exc
        try:
            data = json.loads(raw)
        except Exception as exc:
            raise DataWorksApiError(f'无法解析 DataWorks 响应: {raw[:500]}') from exc
        if isinstance(data, dict):
            success = data.get('Success')
            code = data.get('Code') or data.get('HttpStatusCode')
            request_id = _normalize_text(data.get('RequestId'))
            error_message = (
                data.get('Message')
                or data.get('ErrorMessage')
                or data.get('ErrorMsg')
                or data.get('Message')
            )
            if success is False or (code not in (None, 200, '200', 'OK', 'Success') and error_message):
                raise self._build_api_error(
                    action,
                    error_code=_normalize_text(code),
                    request_id=request_id,
                    response_message=_normalize_text(error_message or data),
                )
        return data

    def _request(self, action: str, params: Optional[Dict[str, Any]] = None, method: str = 'POST') -> Dict[str, Any]:
        for retry_index in range(self.cfg.dataworks_max_retries + 1):
            try:
                result = self._request_once(action, params=params, method=method)
                if self.cfg.dataworks_request_interval_seconds:
                    time.sleep(self.cfg.dataworks_request_interval_seconds)
                return result
            except DataWorksApiError as exc:
                if not exc.is_throttling or retry_index >= self.cfg.dataworks_max_retries:
                    raise
                delay = min(
                    self.cfg.dataworks_retry_max_seconds,
                    self.cfg.dataworks_retry_base_seconds * (2 ** retry_index),
                )
                print(
                    f"   ⚠️ DataWorks 接口限流，{delay:.1f}s 后重试 "
                    f"({retry_index + 1}/{self.cfg.dataworks_max_retries}): {exc}"
                )
                time.sleep(delay)
        raise DataWorksApiError(f'{action} 请求失败')

    def list_files(self, page_number: int = 1) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            'PageNumber': page_number,
            'PageSize': min(max(1, DEFAULT_PAGE_SIZE), 100),
            'FileTypes': DEFAULT_FILE_TYPES,
        }
        if self.cfg.dataworks_use_type:
            params['UseType'] = self.cfg.dataworks_use_type
        payload = self._request('ListFiles', params=params)
        files, total_count = _extract_file_list(payload)
        return {
            'files': files,
            'total_count': total_count,
            'page_number': page_number,
            'page_size': params['PageSize'],
            'raw': payload,
        }

    def get_file(self, file_id: Optional[Any] = None, node_id: Optional[Any] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if file_id is not None:
            params['FileId'] = file_id
        if node_id is not None:
            params['NodeId'] = node_id
        payload = self._request('GetFile', params=params)
        return _extract_mapping(payload)

    def list_node_dependencies(self, node_id: Any, page_number: int = 1) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            'Id': node_id,
            'PageNumber': page_number,
            'PageSize': min(max(1, DEFAULT_PAGE_SIZE), 100),
        }
        payload = self._request('ListNodeDependencies', params=params)
        nodes, total_count = _extract_dependency_page(payload)
        return {
            'nodes': nodes,
            'total_count': total_count,
            'page_number': page_number,
            'page_size': params['PageSize'],
            'raw': payload,
        }


class HologresKnowledgeStore:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.table_fqdn = f'{DEFAULT_HOLOGRES_SCHEMA}.{DEFAULT_HOLOGRES_TABLE}'
        self.conn = psycopg2.connect(
            host=cfg.hologres_host,
            port=DEFAULT_HOLOGRES_PORT,
            dbname=cfg.hologres_db,
            user=cfg.hologres_user,
            password=cfg.hologres_password
        )
        self.conn.autocommit = False
        self.verify_schema_ready()

    def _schema_help(self) -> str:
        return (
            f'Hologres 知识表 {self.table_fqdn} 不可用。'
            '请确认数据库中已存在该表，并且当前运行账号已获得 knowledge schema 的 USAGE 权限，'
            '以及该表和相关序列的 SELECT / INSERT / UPDATE / DELETE 权限。'
        )

    def verify_schema_ready(self) -> None:
        sql = """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
            LIMIT 1
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, (DEFAULT_HOLOGRES_SCHEMA, DEFAULT_HOLOGRES_TABLE))
                exists = cursor.fetchone() is not None
        except psycopg2.Error as exc:
            self.conn.rollback()
            raise RuntimeError(f'无法检查 Hologres 知识表状态：{exc}') from exc
        if not exists:
            raise RuntimeError(self._schema_help())

    def existing_snapshot(self) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
        snapshot: Dict[str, Tuple[Optional[str], Optional[int]]] = {}
        sql = f"""
            SELECT node_key, text_hash, id
            FROM {self.table_fqdn}
            WHERE db_name = %s
              AND project_id = %s
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (DEFAULT_KNOWLEDGE_DB_NAME, self.cfg.project_id or 0))
            for node_key, text_hash, row_id in cursor.fetchall():
                snapshot[str(node_key)] = (text_hash, row_id)
        return snapshot

    def deactivate_missing(self, missing_keys: Sequence[str]) -> int:
        keys = [key for key in missing_keys if _normalize_text(key)]
        if not keys:
            return 0
        sql = f"""
            UPDATE {self.table_fqdn}
            SET is_active = FALSE,
                updated_at = NOW()
            WHERE db_name = %s
              AND project_id = %s
              AND node_key = ANY(%s)
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (DEFAULT_KNOWLEDGE_DB_NAME, self.cfg.project_id or 0, keys))
        return len(keys)

    def upsert_records(self, records: Sequence[Dict[str, Any]]) -> int:
        if not records:
            return 0
        cols = [
            'db_name', 'project_id', 'project_identifier', 'workspace_region',
            'node_key', 'node_id', 'file_id', 'node_name', 'file_name', 'file_folder_path',
            'absolute_folder_path', 'file_type', 'use_type', 'connection_name', 'owner',
            'last_edit_user', 'commit_status', 'auto_parsing', 'is_maxcompute', 'current_version',
            'file_description', 'source_modified_at', 'content', 'input_list', 'output_list',
            'dependent_node_ids', 'upstream_nodes', 'upstream_tables', 'output_tables',
            'node_configuration', 'file_payload',
            'text_hash', 'source_hash', 'last_seen_at', 'is_active', 'updated_at',
        ]
        insert_sql = f"""
            INSERT INTO {self.table_fqdn} (
                {', '.join(cols)}
            ) VALUES %s
            ON CONFLICT (db_name, project_id, node_key) DO UPDATE SET
                project_identifier = EXCLUDED.project_identifier,
                workspace_region = EXCLUDED.workspace_region,
                node_id = EXCLUDED.node_id,
                file_id = EXCLUDED.file_id,
                node_name = EXCLUDED.node_name,
                file_name = EXCLUDED.file_name,
                file_folder_path = EXCLUDED.file_folder_path,
                absolute_folder_path = EXCLUDED.absolute_folder_path,
                file_type = EXCLUDED.file_type,
                use_type = EXCLUDED.use_type,
                connection_name = EXCLUDED.connection_name,
                owner = EXCLUDED.owner,
                last_edit_user = EXCLUDED.last_edit_user,
                commit_status = EXCLUDED.commit_status,
                auto_parsing = EXCLUDED.auto_parsing,
                is_maxcompute = EXCLUDED.is_maxcompute,
                current_version = EXCLUDED.current_version,
                file_description = EXCLUDED.file_description,
                source_modified_at = EXCLUDED.source_modified_at,
                content = EXCLUDED.content,
                input_list = EXCLUDED.input_list,
                output_list = EXCLUDED.output_list,
                dependent_node_ids = EXCLUDED.dependent_node_ids,
                upstream_nodes = EXCLUDED.upstream_nodes,
                upstream_tables = EXCLUDED.upstream_tables,
                output_tables = EXCLUDED.output_tables,
                node_configuration = EXCLUDED.node_configuration,
                file_payload = EXCLUDED.file_payload,
                text_hash = EXCLUDED.text_hash,
                source_hash = EXCLUDED.source_hash,
                last_seen_at = EXCLUDED.last_seen_at,
                is_active = TRUE,
                updated_at = EXCLUDED.updated_at
        """
        rows = []
        for record in records:
            rows.append((
                record['db_name'],
                record['project_id'],
                record.get('project_identifier'),
                record.get('workspace_region'),
                record['node_key'],
                record.get('node_id'),
                record.get('file_id'),
                record.get('node_name'),
                record.get('file_name'),
                record.get('file_folder_path'),
                record.get('absolute_folder_path'),
                record.get('file_type'),
                record.get('use_type'),
                record.get('connection_name'),
                record.get('owner'),
                record.get('last_edit_user'),
                record.get('commit_status'),
                record.get('auto_parsing'),
                record.get('is_maxcompute'),
                record.get('current_version'),
                record.get('file_description'),
                record.get('source_modified_at'),
                record.get('content'),
                Json(record.get('input_list') or [], dumps=_json_dumps),
                Json(record.get('output_list') or [], dumps=_json_dumps),
                Json(record.get('dependent_node_ids') or [], dumps=_json_dumps),
                Json(record.get('upstream_nodes') or [], dumps=_json_dumps),
                Json(record.get('upstream_tables') or [], dumps=_json_dumps),
                Json(record.get('output_tables') or [], dumps=_json_dumps),
                Json(record.get('node_configuration') or {}, dumps=_json_dumps),
                Json(record.get('file_payload') or {}, dumps=_json_dumps),
                record.get('text_hash'),
                record.get('source_hash'),
                record.get('last_seen_at'),
                True,
                record.get('updated_at'),
            ))

        with self.conn.cursor() as cursor:
            execute_values(cursor, insert_sql, rows, page_size=100)
        self.conn.commit()
        return len(rows)


class DataWorksSyncJob:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.client = DataWorksOpenAPIClient(cfg)
        self.store = HologresKnowledgeStore(cfg)

    def _fetch_all_files(self) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        page_number = 1
        while page_number <= DEFAULT_MAX_PAGES:
            payload = self.client.list_files(page_number=page_number)
            page_files = payload.get('files') or []
            files.extend(page_files)
            total_count = payload.get('total_count') or len(files)
            if len(files) >= total_count or len(page_files) < min(max(1, DEFAULT_PAGE_SIZE), 100):
                break
            page_number += 1
        return files

    def _merge_file_records(self, list_file: Dict[str, Any], detail_file: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        merged = dict(list_file or {})
        if detail_file:
            for key, value in detail_file.items():
                if key not in merged or merged[key] in (None, ''):
                    merged[key] = value

        node_id = _normalize_text(merged.get('NodeId') or merged.get('node_id') or merged.get('NodeID'))
        file_id = _normalize_text(merged.get('FileId') or merged.get('file_id') or merged.get('FileID'))
        if not node_id and not file_id:
            return None

        node_key = node_id or file_id
        node_config = _extract_node_config(merged)
        content = _extract_content(merged)
        input_list = _normalize_list(node_config.get('InputList') or node_config.get('inputList') or node_config.get('Inputs'))
        output_list = _normalize_list(node_config.get('OutputList') or node_config.get('outputList') or node_config.get('Outputs'))
        dependent_node_ids = _split_ids(
            node_config.get('DependentNodeIdList')
            or node_config.get('dependentNodeIdList')
            or node_config.get('DependentNodeIds')
            or node_config.get('dependentNodeIds')
        )
        output_tables = _extract_table_names(output_list)
        upstream_tables = _extract_table_names(input_list)

        source_payload = {
            'file': merged,
            'node_config': node_config,
            'content': content,
            'input_list': input_list,
            'output_list': output_list,
            'dependent_node_ids': dependent_node_ids,
            'output_tables': output_tables,
            'upstream_tables': upstream_tables,
        }

        raw_description = _normalize_text(
            merged.get('FileDescription')
            or merged.get('file_description')
            or merged.get('Description')
            or merged.get('description')
        )
        node_name = _normalize_text(
            merged.get('FileName')
            or merged.get('file_name')
            or merged.get('NodeName')
            or merged.get('node_name')
            or merged.get('Name')
            or merged.get('name')
        )
        file_folder_path = _normalize_text(
            merged.get('FileFolderPath')
            or merged.get('file_folder_path')
            or merged.get('FolderPath')
            or merged.get('folder_path')
        )
        absolute_folder_path = _normalize_text(
            merged.get('AbsoluteFolderPath')
            or merged.get('absolute_folder_path')
            or file_folder_path
        )
        connection_name = _normalize_text(merged.get('ConnectionName') or merged.get('connection_name'))
        owner = _normalize_text(merged.get('Owner') or merged.get('owner'))
        last_edit_user = _normalize_text(merged.get('LastEditUser') or merged.get('last_edit_user'))
        file_type = _normalize_text(merged.get('FileType') or merged.get('file_type'))
        use_type = _normalize_text(merged.get('UseType') or merged.get('use_type'))
        commit_status = _normalize_text(merged.get('CommitStatus') or merged.get('commit_status'))
        current_version = _normalize_text(merged.get('CurrentVersion') or merged.get('current_version'))
        source_modified_at = _parse_datetime(
            merged.get('LastEditTime')
            or merged.get('last_edit_time')
            or merged.get('ModifiedTime')
            or merged.get('modified_time')
        )
        auto_parsing = _normalize_bool(merged.get('AutoParsing') or merged.get('auto_parsing'))
        is_maxcompute = _normalize_bool(merged.get('IsMaxCompute') or merged.get('is_maxcompute'))
        source_hash = _safe_json_hash(source_payload)
        text_hash = _safe_json_hash({
            'node_name': node_name,
            'file_name': merged.get('FileName') or merged.get('file_name'),
            'absolute_folder_path': absolute_folder_path,
            'content': content,
            'upstream_tables': upstream_tables,
            'output_tables': output_tables,
            'dependent_node_ids': dependent_node_ids,
        })
        return {
            'db_name': DEFAULT_KNOWLEDGE_DB_NAME,
            'project_id': self.cfg.project_id or 0,
            'project_identifier': '',
            'workspace_region': DEFAULT_REGION_ID,
            'node_key': node_key,
            'node_id': node_id,
            'file_id': file_id,
            'node_name': node_name or node_key,
            'file_name': _normalize_text(merged.get('FileName') or merged.get('file_name')) or node_name or node_key,
            'file_folder_path': file_folder_path,
            'absolute_folder_path': absolute_folder_path,
            'file_type': file_type,
            'use_type': use_type,
            'connection_name': connection_name,
            'owner': owner,
            'last_edit_user': last_edit_user,
            'commit_status': commit_status,
            'auto_parsing': auto_parsing,
            'is_maxcompute': is_maxcompute,
            'current_version': current_version,
            'file_description': raw_description,
            'source_modified_at': source_modified_at,
            'content': content,
            'input_list': input_list,
            'output_list': output_list,
            'dependent_node_ids': dependent_node_ids,
            'upstream_nodes': [],
            'upstream_tables': upstream_tables,
            'output_tables': output_tables,
            'node_configuration': node_config,
            'file_payload': merged,
            'text_hash': text_hash,
            'source_hash': source_hash,
            'last_seen_at': datetime.now(),
            'updated_at': datetime.now(),
        }

    def _attach_dependency_nodes(self, records: List[Dict[str, Any]]) -> None:
        node_map: Dict[str, Dict[str, Any]] = {}
        for record in records:
            for key in (record.get('node_key'), record.get('node_id')):
                key_text = _normalize_text(key)
                if key_text:
                    node_map[key_text] = record

        for record in records:
            node_id = _normalize_text(record.get('node_id'))
            if not node_id:
                continue
            try:
                dependencies: List[Dict[str, Any]] = []
                page_number = 1
                while page_number <= DEFAULT_MAX_PAGES:
                    payload = self.client.list_node_dependencies(node_id=node_id, page_number=page_number)
                    page_nodes = payload.get('nodes') or []
                    dependencies.extend(page_nodes)
                    total_count = payload.get('total_count') or len(dependencies)
                    if len(dependencies) >= total_count or len(page_nodes) < min(max(1, DEFAULT_PAGE_SIZE), 100):
                        break
                    page_number += 1
                if not dependencies:
                    continue
            except Exception as exc:
                raise RuntimeError(f"获取 DataWorks 节点依赖失败：node_id={node_id}。{exc}") from exc

            upstream_nodes = []
            upstream_names = []
            upstream_tables = list(record.get('upstream_tables') or [])
            dependent_node_ids = list(dict.fromkeys(_split_ids(record.get('dependent_node_ids'))))

            for dep in dependencies:
                dep_norm = _normalize_dependency_node(dep)
                dep_key = _normalize_text(dep_norm.get('node_key') or dep_norm.get('node_id'))
                if not dep_key:
                    continue
                dep_record = node_map.get(dep_key) or node_map.get(_normalize_text(dep_norm.get('node_id')))
                if dep_record:
                    dep_norm['node_name'] = dep_record.get('node_name') or dep_norm.get('node_name') or dep_key
                    dep_norm['file_name'] = dep_record.get('file_name') or dep_norm.get('file_name') or dep_key
                    dep_norm['absolute_folder_path'] = dep_record.get('absolute_folder_path') or dep_norm.get('absolute_folder_path') or ''
                    dep_norm['output_tables'] = list(dict.fromkeys(list(dep_norm.get('output_tables') or []) + list(dep_record.get('output_tables') or [])))
                upstream_nodes.append(dep_norm)
                name = _normalize_text(dep_norm.get('node_name') or dep_norm.get('file_name') or dep_key)
                if name:
                    upstream_names.append(name)
                upstream_tables.extend(dep_norm.get('output_tables') or [])
                dependent_node_ids.append(dep_key)

            upstream_names = list(dict.fromkeys([name for name in upstream_names if name]))
            upstream_tables = list(dict.fromkeys([table for table in upstream_tables if table]))
            record['dependent_node_ids'] = list(dict.fromkeys([x for x in dependent_node_ids if x]))
            record['upstream_nodes'] = upstream_nodes
            record['upstream_tables'] = upstream_tables
            record['text_hash'] = _safe_json_hash({
                'node_name': record.get('node_name') or '',
                'file_name': record.get('file_name') or '',
                'absolute_folder_path': record.get('absolute_folder_path') or '',
                'content': record.get('content') or '',
                'dependent_node_ids': record.get('dependent_node_ids') or [],
                'upstream_nodes': record.get('upstream_nodes') or [],
                'upstream_tables': record.get('upstream_tables') or [],
                'output_tables': record.get('output_tables') or [],
            })
            record['source_hash'] = _safe_json_hash({
                'file': record.get('file_payload') or {},
                'node_config': record.get('node_configuration') or {},
                'content': record.get('content') or '',
                'input_list': record.get('input_list') or [],
                'output_list': record.get('output_list') or [],
                'dependent_node_ids': record.get('dependent_node_ids') or [],
                'upstream_nodes': record.get('upstream_nodes') or [],
                'upstream_tables': record.get('upstream_tables') or [],
                'output_tables': record.get('output_tables') or [],
            })

    def _get_sync_stats(self, records: Sequence[Dict[str, Any]], snapshot: Dict[str, Tuple[Optional[str], Optional[int]]]) -> Tuple[Dict[str, int], List[str]]:
        stats = {'new': 0, 'changed': 0, 'unchanged': 0, 'removed': 0}
        current_keys = []
        for record in records:
            key = str(record.get('node_key') or '')
            if not key:
                continue
            current_keys.append(key)
            old_hash, _row_id = snapshot.get(key, (None, None))
            text_hash = record.get('text_hash')
            if old_hash is None:
                stats['new'] += 1
            elif old_hash != text_hash:
                stats['changed'] += 1
            else:
                stats['unchanged'] += 1
        removed_keys = sorted(set(snapshot.keys()) - set(current_keys))
        stats['removed'] = len(removed_keys)
        return stats, removed_keys

    def run(self) -> Dict[str, Any]:
        if not self.client.is_configured():
            raise RuntimeError('请在 MANUAL_CONFIG 里配置 DATAWORKS_ACCESS_KEY_ID / DATAWORKS_ACCESS_KEY_SECRET / DATAWORKS_PROJECT_ID')
        if not self.cfg.hologres_host or not self.cfg.hologres_db or not self.cfg.hologres_user or not self.cfg.hologres_password:
            raise RuntimeError(
                '请在 MANUAL_CONFIG 里配置 Hologres 连接信息：HOLOGRES_HOST / HOLOGRES_DB / HOLOGRES_USER / HOLOGRES_PASSWORD'
            )

        files = self._fetch_all_files()
        records: List[Dict[str, Any]] = []
        for raw_file in files:
            node_id = _normalize_text(raw_file.get('NodeId') or raw_file.get('node_id'))
            file_id = _normalize_text(raw_file.get('FileId') or raw_file.get('file_id'))
            file_name = _normalize_text(
                raw_file.get('FileName')
                or raw_file.get('file_name')
                or raw_file.get('NodeName')
                or raw_file.get('node_name')
            )
            if not file_id and not node_id:
                raise RuntimeError(
                    f"ListFiles 返回的文件缺少 FileId/NodeId，无法调用 GetFile：file_name={file_name or '<unknown>'}"
                )
            try:
                detail = self.client.get_file(file_id=file_id) if file_id else self.client.get_file(node_id=node_id)
            except Exception as exc:
                raise RuntimeError(
                    f"获取 DataWorks 文件详情失败：file_id={file_id or '<empty>'}, "
                    f"node_id={node_id or '<empty>'}, file_name={file_name or '<unknown>'}。{exc}"
                ) from exc
            record = self._merge_file_records(raw_file, detail)
            if record:
                records.append(record)

        self._attach_dependency_nodes(records)

        snapshot = self.store.existing_snapshot()
        stats, removed_keys = self._get_sync_stats(records, snapshot)

        print(f"   ├─ 节点总数: {len(records)}")
        print(f"   ├─ 新增 {stats['new']}, 修改 {stats['changed']}, 不变 {stats['unchanged']}, 移除 {stats['removed']}")

        current_ts = datetime.now()
        for record in records:
            record['last_seen_at'] = current_ts
            record['updated_at'] = current_ts

        if removed_keys:
            self.store.deactivate_missing(removed_keys)

        if records:
            self.store.upsert_records(records)

        result = {
            'ok': True,
            'enabled': True,
            'project_id': self.cfg.project_id,
            'files': len(files),
            'processed': len(records),
            'stats': stats,
            'removed_keys': removed_keys[:50],
            'sync_mode': 'content_only',
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
        return result

def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            return None
    text_value = str(value).strip()
    if not text_value:
        return None
    for fmt in (
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%d',
    ):
        try:
            return datetime.strptime(text_value[:len(fmt)], fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(text_value.replace('Z', '+00:00')).replace(tzinfo=None)
    except Exception:
        return None


def main() -> int:
    cfg = RuntimeConfig.from_argv(sys.argv[1:])
    if cfg.dry_run:
        safe_cfg = dict(cfg.__dict__)
        if safe_cfg.get('access_key_secret'):
            safe_cfg['access_key_secret'] = '***'
        if safe_cfg.get('hologres_password'):
            safe_cfg['hologres_password'] = '***'
        print(json.dumps(safe_cfg, ensure_ascii=False, indent=2, default=_json_default))
        return 0

    job = DataWorksSyncJob(cfg)
    job.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
