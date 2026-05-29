# core/cancellation.py - 查询取消管理
import threading
import uuid
from typing import Dict, Optional


class CancelledError(Exception):
    """查询被用户取消时抛出"""
    pass


class CancellationToken:
    """单个查询的取消令牌"""
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self):
        """检查点：被取消则抛异常，调用方在每个步骤间调用"""
        if self._event.is_set():
            raise CancelledError("查询已被用户取消")


class _CancellationRegistry:
    """全局取消令牌注册表，按 request_id 索引"""
    def __init__(self):
        self._tokens: Dict[str, CancellationToken] = {}
        self._lock = threading.Lock()

    def create(self) -> tuple:
        """创建新 token，返回 (request_id, token)"""
        request_id = uuid.uuid4().hex
        token = CancellationToken()
        with self._lock:
            self._tokens[request_id] = token
        return request_id, token

    def register_with_id(self, request_id: str) -> CancellationToken:
        """用前端给的 id 注册 token"""
        token = CancellationToken()
        with self._lock:
            self._tokens[request_id] = token
        return token

    def cancel(self, request_id: str) -> bool:
        """取消指定 request_id 对应的查询；返回是否找到"""
        with self._lock:
            token = self._tokens.get(request_id)
        if token is None:
            return False
        token.cancel()
        return True

    def cleanup(self, request_id: str):
        """请求结束后清理 token（无论成功、失败、取消）"""
        with self._lock:
            self._tokens.pop(request_id, None)


# 全局单例
registry = _CancellationRegistry()
