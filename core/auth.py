# core/auth.py - 登录认证与当前用户上下文
import hashlib
import hmac
import os
import uuid
from typing import Dict, Optional

from flask import g, has_request_context, request, session
from sqlalchemy import text
from werkzeug.security import check_password_hash, generate_password_hash

from .db_manager import DatabasePoolManager


SESSION_ADMIN_ID_KEY = "admin_id"
SESSION_ID_KEY = "session_id"
SESSION_USER_KEY = "user_snapshot"


def get_auth_db_name() -> str:
    return os.getenv("AUTH_DB_NAME", os.getenv("APP_AUTH_DB_NAME", "hologres"))


def _get_auth_engine():
    return DatabasePoolManager.get_engine(get_auth_db_name())


def _row_to_user(row) -> Optional[Dict]:
    if not row:
        return None
    data = dict(row._mapping)
    if data.get("admin_id") is not None:
        data["admin_id"] = int(data["admin_id"])
    return data


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _bootstrap_admin_identity() -> Dict:
    admin_id = _safe_int(os.getenv("APP_ADMIN_ID"), 1)
    account = os.getenv("APP_ADMIN_ACCOUNT", "admin").strip() or "admin"
    user_name = os.getenv("APP_ADMIN_USER_NAME", "admin").strip() or "admin"
    mobile = os.getenv("APP_ADMIN_MOBILE", "").strip() or None
    role_id = _safe_int(os.getenv("APP_ADMIN_ROLE_ID"), 1)
    role_name = os.getenv("APP_ADMIN_ROLE_NAME", "超级管理员").strip() or "超级管理员"
    return {
        "admin_id": admin_id,
        "user_name": user_name,
        "mobile": mobile,
        "we_user_id": account,
        "employee_id": account,
        "role_id": role_id,
        "role_name": role_name,
        "admin_organ_id": None,
        "organ_name": "系统管理员",
        "teacher_uid": None,
        "subject": None,
        "status": 1,
        "is_full_view": 1,
        "permission_type": 1,
        "permission_scope": 1,
        "admin_organ_ids": None,
        "parent_ids": None,
    }


def _bootstrap_admin_login_matches(login_name: str) -> bool:
    login = str(login_name or "").strip()
    if not login:
        return False

    identity = _bootstrap_admin_identity()
    candidates = {
        str(identity["admin_id"]),
        str(os.getenv("APP_ADMIN_ACCOUNT", "admin")).strip(),
        str(os.getenv("APP_ADMIN_USER_NAME", "admin")).strip(),
        str(os.getenv("APP_ADMIN_MOBILE", "")).strip(),
    }
    candidates = {item for item in candidates if item}
    return login in candidates


def _verify_password_hash(stored_hash: str, password: str) -> bool:
    candidate = password or ""
    stored = str(stored_hash or "").strip()
    if not stored:
        return False

    try:
        if check_password_hash(stored, candidate):
            return True
    except Exception:
        pass

    lowered = stored.lower()
    try:
        if lowered.startswith(("$2a$", "$2b$", "$2y$")):
            import bcrypt

            return bool(bcrypt.checkpw(candidate.encode("utf-8"), stored.encode("utf-8")))
    except Exception:
        pass

    hex_chars = set("0123456789abcdef")
    if len(lowered) == 32 and set(lowered) <= hex_chars:
        return hmac.compare_digest(hashlib.md5(candidate.encode("utf-8")).hexdigest(), lowered)
    if len(lowered) == 40 and set(lowered) <= hex_chars:
        return hmac.compare_digest(hashlib.sha1(candidate.encode("utf-8")).hexdigest(), lowered)
    if len(lowered) == 64 and set(lowered) <= hex_chars:
        return hmac.compare_digest(hashlib.sha256(candidate.encode("utf-8")).hexdigest(), lowered)

    return hmac.compare_digest(stored, candidate)


def user_can_view_all(user: Optional[Dict]) -> bool:
    if not user:
        return False

    admin_ids = {
        item.strip()
        for item in os.getenv("APP_SUPER_ADMIN_IDS", "").split(",")
        if item.strip()
    }
    if str(user.get("admin_id")) in admin_ids:
        return True

    role_ids = {
        item.strip()
        for item in os.getenv("APP_ADMIN_ROLE_IDS", "1").split(",")
        if item.strip()
    }
    if str(user.get("role_id") or "") in role_ids:
        return True

    if _safe_int(user.get("is_full_view")) == 1:
        return True

    role_name = str(user.get("role_name") or "").lower()
    return "admin" in role_name or "管理员" in role_name or "超级" in role_name


def bootstrap_admin_user() -> bool:
    """用环境变量引导一个 admin 账号；未配置密码时不创建，避免硬编码弱密码。"""
    password = os.getenv("APP_ADMIN_PASSWORD")
    password_hash = os.getenv("APP_ADMIN_PASSWORD_HASH")
    if not password and not password_hash:
        return False

    admin_id = _safe_int(os.getenv("APP_ADMIN_ID"), 1)
    account = os.getenv("APP_ADMIN_ACCOUNT", "admin").strip() or "admin"
    user_name = os.getenv("APP_ADMIN_USER_NAME", "admin").strip() or "admin"
    mobile = os.getenv("APP_ADMIN_MOBILE", "").strip() or None
    role_id = _safe_int(os.getenv("APP_ADMIN_ROLE_ID"), 1)
    role_name = os.getenv("APP_ADMIN_ROLE_NAME", "超级管理员").strip() or "超级管理员"
    final_hash = password_hash or generate_password_hash(password)

    sql = """
    INSERT INTO knowledge.app_user (
        admin_id, user_name, mobile, we_user_id, employee_id,
        role_id, role_name, admin_organ_id, organ_name, teacher_uid, subject,
        status, is_full_view, permission_type, permission_scope,
        admin_organ_ids, parent_ids, password_hash, created_at, updated_at
    ) VALUES (
        :admin_id, :user_name, :mobile, :account, :account,
        :role_id, :role_name, NULL, '系统管理员', NULL, NULL,
        1, 1, 1, 1, NULL, NULL, :password_hash, NOW(), NOW()
    )
    ON CONFLICT (admin_id) DO UPDATE SET
        user_name = EXCLUDED.user_name,
        mobile = COALESCE(EXCLUDED.mobile, knowledge.app_user.mobile),
        we_user_id = EXCLUDED.we_user_id,
        employee_id = EXCLUDED.employee_id,
        role_id = EXCLUDED.role_id,
        role_name = EXCLUDED.role_name,
        organ_name = EXCLUDED.organ_name,
        status = 1,
        is_full_view = 1,
        permission_type = 1,
        permission_scope = 1,
        password_hash = EXCLUDED.password_hash,
        updated_at = NOW()
    """

    try:
        with _get_auth_engine().connect() as conn:
            conn.execute(
                text(sql),
                {
                    "admin_id": admin_id,
                    "user_name": user_name,
                    "mobile": mobile,
                    "account": account,
                    "role_id": role_id,
                    "role_name": role_name,
                    "password_hash": final_hash,
                },
            )
            conn.commit()
        print(f"   ✅ admin 引导账号已同步: {account} (admin_id={admin_id})")
        return True
    except Exception as e:
        safe_message = str(e).splitlines()[0]
        print(f"   ⚠️ admin 引导账号同步失败: {type(e).__name__}: {safe_message}")
        return False


def find_user_by_login(login_name: str) -> Optional[Dict]:
    login = str(login_name or "").strip()
    if not login:
        return None

    sql = """
    SELECT admin_id, user_name, mobile, we_user_id, employee_id,
           role_id, role_name, admin_organ_id, organ_name, teacher_uid, subject,
           status, is_full_view, permission_type, permission_scope,
           admin_organ_ids, parent_ids, password_hash, last_login_at,
           created_at, updated_at
    FROM knowledge.app_user
    WHERE COALESCE(status, 1) <> 0
      AND (
          CAST(admin_id AS TEXT) = :login
          OR mobile = :login
          OR we_user_id = :login
          OR employee_id = :login
          OR user_name = :login
      )
    ORDER BY updated_at DESC
    LIMIT 1
    """
    with _get_auth_engine().connect() as conn:
        return _row_to_user(conn.execute(text(sql), {"login": login}).fetchone())


def authenticate_user(login_name: str, password: str) -> Optional[Dict]:
    user = find_user_by_login(login_name)
    if user:
        password_hash = user.get("password_hash")
        if password_hash and _verify_password_hash(password_hash, password or ""):
            try:
                with _get_auth_engine().connect() as conn:
                    conn.execute(
                        text("""
                            UPDATE knowledge.app_user
                            SET last_login_at = NOW(), updated_at = NOW()
                            WHERE admin_id = :admin_id
                        """),
                        {"admin_id": user["admin_id"]},
                    )
                    conn.commit()
            except Exception as e:
                print(f"   ⚠️ 更新 last_login_at 失败: {e}")

            user.pop("password_hash", None)
            return user

    if _bootstrap_admin_login_matches(login_name):
        bootstrap_password = os.getenv("APP_ADMIN_PASSWORD")
        bootstrap_password_hash = os.getenv("APP_ADMIN_PASSWORD_HASH")
        if bootstrap_password_hash:
            password_ok = _verify_password_hash(bootstrap_password_hash, password or "")
        else:
            password_ok = hmac.compare_digest(str(bootstrap_password or ""), str(password or ""))

        if password_ok:
            try:
                bootstrap_admin_user()
            except Exception as e:
                print(f"   ⚠️ admin 兜底同步失败: {e}")
            fallback_user = _bootstrap_admin_identity()
            fallback_user.pop("password_hash", None)
            return fallback_user

    return None


def get_or_create_session_id() -> Optional[str]:
    if not has_request_context():
        return None
    sid = session.get(SESSION_ID_KEY)
    if not sid:
        sid = uuid.uuid4().hex
        session[SESSION_ID_KEY] = sid
    return sid


def login_user(user: Dict):
    session.clear()
    session[SESSION_ADMIN_ID_KEY] = user.get("admin_id")
    session[SESSION_ID_KEY] = uuid.uuid4().hex
    session[SESSION_USER_KEY] = {
        key: user.get(key)
        for key in (
            "admin_id", "user_name", "mobile", "we_user_id", "employee_id",
            "role_id", "role_name", "admin_organ_id", "organ_name",
            "teacher_uid", "subject", "is_full_view", "permission_type",
            "permission_scope", "admin_organ_ids", "parent_ids",
        )
    }
    session.permanent = True


def logout_user():
    session.clear()


def get_current_user(silent: bool = True) -> Optional[Dict]:
    if not has_request_context():
        return None

    if hasattr(g, "current_user"):
        return g.current_user

    admin_id = session.get(SESSION_ADMIN_ID_KEY)
    if not admin_id:
        g.current_user = None
        return None

    try:
        with _get_auth_engine().connect() as conn:
            row = conn.execute(
                text("""
                    SELECT admin_id, user_name, mobile, we_user_id, employee_id,
                           role_id, role_name, admin_organ_id, organ_name,
                           teacher_uid, subject, status, is_full_view,
                           permission_type, permission_scope, admin_organ_ids,
                           parent_ids, last_login_at, created_at, updated_at
                    FROM knowledge.app_user
                    WHERE admin_id = :admin_id AND COALESCE(status, 1) <> 0
                    LIMIT 1
                """),
                {"admin_id": admin_id},
            ).fetchone()
            user = _row_to_user(row)
    except Exception as e:
        if not silent:
            raise
        print(f"   ⚠️ 获取当前用户失败，使用 session 快照: {e}")
        user = session.get(SESSION_USER_KEY)

    if user:
        user.pop("password_hash", None)
    g.current_user = user
    return user


def get_client_ip() -> str:
    if not has_request_context():
        return ""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.headers.get("X-Real-IP") or request.remote_addr or ""


def build_log_context(action_type: str, request_id: Optional[str] = None,
                      visibility_scope: Optional[str] = None) -> Dict:
    user = get_current_user()
    scope = visibility_scope
    if not scope:
        scope = 'all' if user_can_view_all(user) else 'self'
    return {
        "admin_id": user.get("admin_id") if user else None,
        "request_id": request_id,
        "session_id": get_or_create_session_id(),
        "client_ip": get_client_ip(),
        "user_agent": request.headers.get("User-Agent", "")[:1000] if has_request_context() else "",
        "action_type": action_type,
        "visibility_scope": scope,
    }


def public_user_payload(user: Optional[Dict]) -> Optional[Dict]:
    if not user:
        return None
    return {
        "admin_id": user.get("admin_id"),
        "user_name": user.get("user_name"),
        "mobile": user.get("mobile"),
        "we_user_id": user.get("we_user_id"),
        "employee_id": user.get("employee_id"),
        "role_id": user.get("role_id"),
        "role_name": user.get("role_name"),
        "admin_organ_id": user.get("admin_organ_id"),
        "organ_name": user.get("organ_name"),
        "can_view_all": user_can_view_all(user),
    }
