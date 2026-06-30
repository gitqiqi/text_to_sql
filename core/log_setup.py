import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_MAIN_LOG = _LOG_DIR / "app.log"
_AI_LOG = _LOG_DIR / "aisql.log"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def _make_handler(log_path: str, level: int = logging.INFO) -> RotatingFileHandler:
    h = RotatingFileHandler(
        log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8",
    )
    h.setLevel(level)
    h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    return h


def setup_logging(app: "Flask") -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 主日志 handler
    main_handler = _make_handler(str(_MAIN_LOG))

    # 控制台 handler
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S",
    ))

    # 根 logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in (main_handler, stream):
        root.addHandler(h)

    # werkzeug 请求日志
    for h in (main_handler, stream):
        logging.getLogger("werkzeug").handlers.clear()
        logging.getLogger("werkzeug").addHandler(h)
        logging.getLogger("werkzeug").setLevel(logging.INFO)

    # 各模块 logger
    for name in ("knowledge", "db_manager", "query_log", "llm_client"):
        logging.getLogger(name).setLevel(logging.INFO)

    # ---------- AI SQL 专用日志 ----------
    ai_handler = _make_handler(str(_AI_LOG))
    aisql_logger = logging.getLogger("aisql")
    aisql_logger.setLevel(logging.INFO)
    aisql_logger.handlers.clear()
    aisql_logger.addHandler(ai_handler)
    aisql_logger.addHandler(stream)  # 也打到控制台

    app.logger.handlers.clear()
    for h in (main_handler, stream):
        app.logger.addHandler(h)

    app.logger.info("=" * 60)
    app.logger.info("Text2SQL 应用启动")
    app.logger.info("主日志: %s", _MAIN_LOG)
    app.logger.info("AI SQL 日志: %s", _AI_LOG)
    app.logger.info("=" * 60)
