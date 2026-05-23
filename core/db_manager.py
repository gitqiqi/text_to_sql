# core/db_manager.py - 数据库连接池与执行器
import threading
from typing import Dict
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text

from config import get_database_config
from .utils import (
    POOL_SIZE,
    MAX_OVERFLOW,
    POOL_PRE_PING,
    monitor_function,
    clean_sql,
    validate_sql_safety,
)


class DatabasePoolManager:
    _engines = {}
    _lock = threading.Lock()

    @classmethod
    def get_engine(cls, db_name: str):
        if db_name not in cls._engines:
            with cls._lock:
                if db_name not in cls._engines:
                    db_config = get_database_config(db_name)
                    if not db_config:
                        raise ValueError(f"数据库配置不存在: {db_name}")
                    cls._engines[db_name] = cls._create_engine(db_config)
                    print(f"🔌 为数据库 {db_name} 创建连接池")
        return cls._engines[db_name]

    @staticmethod
    def _create_engine(db_config: Dict):
        db_type = db_config['type']

        if db_type == 'postgresql':
            user = quote_plus(str(db_config['user']))
            password = quote_plus(str(db_config['password']))
            host = db_config['host']
            port = db_config.get('port', '5432')
            name = quote_plus(str(db_config['name']))
            sslmode = quote_plus(str(db_config.get('sslmode', 'prefer')))
            url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}?sslmode={sslmode}"

            engine = create_engine(
                url,
                pool_size=POOL_SIZE,
                max_overflow=MAX_OVERFLOW,
                pool_pre_ping=POOL_PRE_PING,
                pool_recycle=3600,
                echo=False
            )
            return engine

        elif db_type == 'sqlite':
            url = f"sqlite:///{db_config['file_path']}"
            return create_engine(url, pool_size=1, pool_recycle=3600)

        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")


class DatabaseManager:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.engine = DatabasePoolManager.get_engine(db_name)

    @monitor_function
    def execute_sql(self, sql: str) -> pd.DataFrame:
        if not sql:
            raise ValueError("SQL语句为空")

        cleaned = clean_sql(sql)
        if not cleaned:
            raise ValueError("SQL语句为空")

        if not cleaned.lower().startswith(('select', 'with')):
            raise ValueError("只允许SELECT查询")

        validate_sql_safety(cleaned)

        with self.engine.connect() as conn:
            if self.db_name == 'hologres':
                try:
                    conn.execute(text("SET hg_computing_resource = 'serverless'"))
                except Exception:
                    pass

            result = conn.execute(text(cleaned))
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))

            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
        return df
