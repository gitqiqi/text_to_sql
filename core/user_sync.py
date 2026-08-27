# core/user_sync.py - 账号资料与密码同步
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import text
from werkzeug.security import generate_password_hash

from .db_manager import DatabasePoolManager


DEFAULT_SOURCE_TABLE = "bi.dim_org_admin_user_info_hf"
DEFAULT_PASSWORD_COLUMNS: Tuple[str, ...] = (
    "password_hash",
    "password",
    "passwd",
    "pwd",
    "login_password",
    "login_pwd",
)
DEFAULT_BATCH_SIZE = 500
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HEX_HASH_RE = re.compile(r"^[0-9a-fA-F]+$")


_monitor_thread = None
_monitor_lock = threading.Lock()
_sync_lock = threading.Lock()
_last_sync_result: Optional[Dict[str, Any]] = None


def _auth_db_name() -> str:
    return os.getenv("AUTH_DB_NAME", os.getenv("APP_AUTH_DB_NAME", "hologres"))


def _bool_env(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None and raw.strip() else default
    except Exception:
        return default


def _user_sync_interval_minutes() -> int:
    return max(1, _int_env("APP_USER_SYNC_INTERVAL_MINUTES", 60))


def _parse_candidates(raw_value: Optional[str], fallback: Sequence[str]) -> List[str]:
    if raw_value:
        items = [item.strip() for item in raw_value.split(",")]
        items = [item for item in items if item]
        if items:
            return items
    return list(fallback)


def _split_relation_name(relation: str) -> Tuple[str, str]:
    raw = str(relation or "").strip()
    if not raw:
        raise ValueError("APP_USER_SYNC_SOURCE_TABLE 不能为空")
    if "." in raw:
        schema, table = raw.split(".", 1)
    else:
        schema, table = "public", raw
    schema = schema.strip()
    table = table.strip()
    if not IDENTIFIER_RE.fullmatch(schema) or not IDENTIFIER_RE.fullmatch(table):
        raise ValueError(f"非法的表名配置: {raw}")
    return schema, table


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_relation(schema: str, table: str) -> str:
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


def _build_column_lookup(columns: Sequence[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for column in columns:
        lookup[str(column).lower()] = str(column)
    return lookup


def _resolve_column(column_lookup: Mapping[str, str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        actual = column_lookup.get(candidate.lower())
        if actual:
            return actual
    return None


def _auto_detect_password_column(column_lookup: Mapping[str, str]) -> Optional[str]:
    for actual in column_lookup.values():
        normalized = re.sub(r'[^a-z0-9]+', '', str(actual).lower())
        if not normalized:
            continue
        if (
            'password' in normalized
            or 'passwd' in normalized
            or normalized.endswith('pwd')
            or 'loginpassword' in normalized
            or 'loginpwd' in normalized
            or 'userpassword' in normalized
            or 'passhash' in normalized
            or 'passwordhash' in normalized
            or 'passwordmd5' in normalized
            or 'passmd5' in normalized
        ):
            return actual
    return None


def _first_present(row: Mapping[str, Any], column_lookup: Mapping[str, str], candidates: Sequence[str]) -> Any:
    column = _resolve_column(column_lookup, candidates)
    if not column:
        return None
    return row.get(column)


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text_value = value.strip()
        return text_value or None
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        if not items:
            return None
        return json.dumps(items, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    text_value = str(value).strip()
    return text_value or None


def _normalize_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _looks_like_password_hash(value: str) -> bool:
    if not value:
        return False
    lowered = value.lower()
    if lowered.startswith(("pbkdf2:", "scrypt:", "argon2:", "bcrypt:")):
        return True
    if value.startswith(("$2a$", "$2b$", "$2x$", "$2y$")):
        return True
    if lowered.startswith(("md5:", "sha1:", "sha256:")):
        return True
    if len(value) in {32, 40, 64} and HEX_HASH_RE.fullmatch(value):
        return True
    head = value.split("$", 1)[0]
    return ":" in head and "$" in value


def _normalize_password(raw_value: Any, source_column: Optional[str]) -> Optional[str]:
    normalized = _normalize_text(raw_value)
    if not normalized:
        return None
    column_name = (source_column or "").lower()
    if column_name.endswith("_hash") or column_name.endswith("_digest") or _looks_like_password_hash(normalized):
        return normalized
    return generate_password_hash(normalized)


def _normalize_temporal(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text_value = value.strip()
        return text_value or None
    return value


def _build_user_payload(
    row: Mapping[str, Any],
    column_lookup: Mapping[str, str],
    password_column: Optional[str],
) -> Optional[Dict[str, Any]]:
    admin_id = _normalize_int(_first_present(row, column_lookup, ["admin_id"]))
    if admin_id is None:
        return None

    status = _normalize_int(_first_present(row, column_lookup, ["status"]))
    is_full_view = _normalize_int(_first_present(row, column_lookup, ["is_full_view"]))

    payload = {
        "admin_id": admin_id,
        "user_name": _normalize_text(_first_present(
            row,
            column_lookup,
            ["user_name", "real_name", "name", "nickname", "account", "login_name"],
        )),
        "mobile": _normalize_text(_first_present(
            row,
            column_lookup,
            ["mobile", "phone", "phone_no", "phone_number", "mobile_no"],
        )),
        "we_user_id": _normalize_text(_first_present(
            row,
            column_lookup,
            ["we_user_id", "wechat_user_id", "wechat_id", "wx_user_id", "wecom_user_id"],
        )),
        "employee_id": _normalize_text(_first_present(
            row,
            column_lookup,
            ["employee_id", "employee_no", "work_no", "job_number"],
        )),
        "role_id": _normalize_int(_first_present(row, column_lookup, ["role_id"])),
        "role_name": _normalize_text(_first_present(row, column_lookup, ["role_name"])),
        "admin_organ_id": _normalize_int(_first_present(
            row,
            column_lookup,
            ["admin_organ_id", "organ_id", "org_id", "department_id"],
        )),
        "organ_name": _normalize_text(_first_present(
            row,
            column_lookup,
            ["organ_name", "org_name", "department_name"],
        )),
        "teacher_uid": _normalize_int(_first_present(row, column_lookup, ["teacher_uid", "teacher_id"])),
        "subject": _normalize_int(_first_present(row, column_lookup, ["subject", "subject_id"])),
        "status": 1 if status is None else status,
        "is_full_view": 0 if is_full_view is None else is_full_view,
        "permission_type": _normalize_int(_first_present(row, column_lookup, ["permission_type"])),
        "permission_scope": _normalize_int(_first_present(row, column_lookup, ["permission_scope"])),
        "admin_organ_ids": _normalize_text(_first_present(row, column_lookup, ["admin_organ_ids"])),
        "parent_ids": _normalize_text(_first_present(row, column_lookup, ["parent_ids"])),
        "password_hash": _normalize_password(
            _first_present(row, column_lookup, [password_column]) if password_column else None,
            password_column,
        ),
        "source_create_date": _first_present(
            row,
            column_lookup,
            ["source_create_date", "create_date", "created_at"],
        ),
        "source_update_date": _first_present(
            row,
            column_lookup,
            ["source_update_date", "update_date", "updated_at"],
        ),
    }
    payload["source_create_date"] = _normalize_temporal(payload["source_create_date"])
    payload["source_update_date"] = _normalize_temporal(payload["source_update_date"])
    return payload


def sync_app_users_from_source() -> Dict[str, Any]:
    """从源平台同步 app_user 账号资料与密码。"""
    global _last_sync_result
    with _sync_lock:
        source_table = os.getenv("APP_USER_SYNC_SOURCE_TABLE", DEFAULT_SOURCE_TABLE)
        if not _bool_env("APP_USER_SYNC_ENABLED", True):
            print("   ℹ️ 账号同步已关闭（APP_USER_SYNC_ENABLED=false）")
            result = {
                "ok": True,
                "enabled": False,
                "source_table": source_table,
                "processed": 0,
                "skipped": 0,
                "password_synced": 0,
                "batches": 0,
            }
            _last_sync_result = result
            return result

        password_columns = _parse_candidates(
            os.getenv("APP_USER_SYNC_PASSWORD_COLUMNS"),
            DEFAULT_PASSWORD_COLUMNS,
        )
        batch_size = max(1, _int_env("APP_USER_SYNC_BATCH_SIZE", DEFAULT_BATCH_SIZE))

        schema, table = _split_relation_name(source_table)
        relation_sql = _quote_relation(schema, table)
        sync_engine = DatabasePoolManager.get_engine(_auth_db_name())

        select_sql = text(f"SELECT * FROM {relation_sql} WHERE admin_id IS NOT NULL")
        upsert_sql = text("""
        INSERT INTO knowledge.app_user (
            admin_id, user_name, mobile, we_user_id, employee_id,
            role_id, role_name, admin_organ_id, organ_name, teacher_uid, subject,
            status, is_full_view, permission_type, permission_scope,
            admin_organ_ids, parent_ids, password_hash,
            source_create_date, source_update_date, updated_at
        ) VALUES (
            :admin_id, :user_name, :mobile, :we_user_id, :employee_id,
            :role_id, :role_name, :admin_organ_id, :organ_name, :teacher_uid, :subject,
            :status, :is_full_view, :permission_type, :permission_scope,
            :admin_organ_ids, :parent_ids, :password_hash,
            :source_create_date, :source_update_date, NOW()
        )
        ON CONFLICT (admin_id) DO UPDATE SET
            user_name = COALESCE(EXCLUDED.user_name, knowledge.app_user.user_name),
            mobile = COALESCE(EXCLUDED.mobile, knowledge.app_user.mobile),
            we_user_id = COALESCE(EXCLUDED.we_user_id, knowledge.app_user.we_user_id),
            employee_id = COALESCE(EXCLUDED.employee_id, knowledge.app_user.employee_id),
            role_id = COALESCE(EXCLUDED.role_id, knowledge.app_user.role_id),
            role_name = COALESCE(EXCLUDED.role_name, knowledge.app_user.role_name),
            admin_organ_id = COALESCE(EXCLUDED.admin_organ_id, knowledge.app_user.admin_organ_id),
            organ_name = COALESCE(EXCLUDED.organ_name, knowledge.app_user.organ_name),
            teacher_uid = COALESCE(EXCLUDED.teacher_uid, knowledge.app_user.teacher_uid),
            subject = COALESCE(EXCLUDED.subject, knowledge.app_user.subject),
            status = COALESCE(EXCLUDED.status, knowledge.app_user.status),
            is_full_view = COALESCE(EXCLUDED.is_full_view, knowledge.app_user.is_full_view),
            permission_type = COALESCE(EXCLUDED.permission_type, knowledge.app_user.permission_type),
            permission_scope = COALESCE(EXCLUDED.permission_scope, knowledge.app_user.permission_scope),
            admin_organ_ids = COALESCE(EXCLUDED.admin_organ_ids, knowledge.app_user.admin_organ_ids),
            parent_ids = COALESCE(EXCLUDED.parent_ids, knowledge.app_user.parent_ids),
            password_hash = COALESCE(EXCLUDED.password_hash, knowledge.app_user.password_hash),
            source_create_date = COALESCE(EXCLUDED.source_create_date, knowledge.app_user.source_create_date),
            source_update_date = COALESCE(EXCLUDED.source_update_date, knowledge.app_user.source_update_date),
            updated_at = NOW()
        """)

        processed = 0
        skipped = 0
        password_synced = 0
        batches = 0

        try:
            with sync_engine.begin() as conn:
                result = conn.execute(select_sql)
                column_lookup = _build_column_lookup(result.keys())
                if "admin_id" not in column_lookup:
                    raise ValueError(f"源表 {source_table} 缺少 admin_id 列")

                password_column = _resolve_column(column_lookup, password_columns)
                if not password_column:
                    password_column = _auto_detect_password_column(column_lookup)
                if not password_column:
                    print(f"   ⚠️ 源表 {source_table} 未找到密码列，账号资料会同步，但 password_hash 不会更新")

                rows = result.mappings().all()
                payloads: List[Dict[str, Any]] = []
                for row in rows:
                    payload = _build_user_payload(row, column_lookup, password_column)
                    if not payload:
                        skipped += 1
                        continue
                    processed += 1
                    if payload.get("password_hash"):
                        password_synced += 1
                    payloads.append(payload)

                    if len(payloads) >= batch_size:
                        conn.execute(upsert_sql, payloads)
                        batches += 1
                        payloads.clear()

                if payloads:
                    conn.execute(upsert_sql, payloads)
                    batches += 1

            print(
                f"   ✅ 账号同步完成: {processed} 条，密码同步 {password_synced} 条"
                f"（批次 {batches}，跳过 {skipped} 条）"
            )
            result = {
                "ok": True,
                "enabled": True,
                "source_table": source_table,
                "processed": processed,
                "skipped": skipped,
                "password_synced": password_synced,
                "batches": batches,
            }
            _last_sync_result = result
            return result
        except Exception as e:
            safe_message = str(e).splitlines()[0]
            print(f"   ⚠️ 账号同步失败: {type(e).__name__}: {safe_message}")
            result = {
                "ok": False,
                "enabled": True,
                "source_table": source_table,
                "processed": processed,
                "skipped": skipped,
                "password_synced": password_synced,
                "batches": batches,
                "error": safe_message,
            }
            _last_sync_result = result
            return result


def is_user_sync_monitor_running() -> bool:
    return _monitor_thread is not None and _monitor_thread.is_alive()


def get_last_user_sync_result() -> Optional[Dict[str, Any]]:
    return _last_sync_result


def start_user_sync_monitor() -> Dict[str, Any]:
    """启动后台账号同步线程：先立即同步一次，然后按间隔循环。"""
    interval_minutes = _user_sync_interval_minutes()
    interval_seconds = interval_minutes * 60
    global _monitor_thread, _last_sync_result

    with _monitor_lock:
        if not _bool_env("APP_USER_SYNC_ENABLED", True):
            print("   ℹ️ 账号同步线程未启动（APP_USER_SYNC_ENABLED=false）")
            result = {
                "ok": True,
                "enabled": False,
                "source_table": os.getenv("APP_USER_SYNC_SOURCE_TABLE", DEFAULT_SOURCE_TABLE),
                "processed": 0,
                "skipped": 0,
                "password_synced": 0,
                "batches": 0,
                "scheduled": False,
            }
            _last_sync_result = result
            return result

        if _monitor_thread is not None and _monitor_thread.is_alive():
            print("   ⚠️ 账号同步线程已在运行")
            return _last_sync_result or {
                "ok": True,
                "enabled": True,
                "source_table": os.getenv("APP_USER_SYNC_SOURCE_TABLE", DEFAULT_SOURCE_TABLE),
                "processed": 0,
                "skipped": 0,
                "password_synced": 0,
                "batches": 0,
            }

        def monitor_loop():
            print(f"🔁 账号同步线程启动，每 {interval_minutes} 分钟执行一次")
            while True:
                sync_app_users_from_source()
                time.sleep(interval_seconds)

        _monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name="UserSyncMonitor")
        _monitor_thread.start()
        print(f"   ✅ 账号同步线程已启动（每 {interval_minutes} 分钟自动同步一次）")
        return _last_sync_result or {
            "ok": True,
            "enabled": True,
            "source_table": os.getenv("APP_USER_SYNC_SOURCE_TABLE", DEFAULT_SOURCE_TABLE),
            "processed": 0,
            "skipped": 0,
            "password_synced": 0,
            "batches": 0,
            "scheduled": True,
        }
