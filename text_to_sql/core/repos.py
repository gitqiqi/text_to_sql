# core/repos.py - 知识库（SQL 知识 + 业务名词）的 CRUD 仓储类
import threading
from typing import Dict, List

from sqlalchemy import text

from .db_manager import DatabasePoolManager


class _BaseRepo:
    """共享建表逻辑（每个 db_name 只跑一次）"""

    _table_initialized = set()
    _init_lock = threading.Lock()

    _CREATE_SQL = """
    CREATE SCHEMA IF NOT EXISTS knowledge;
    CREATE TABLE IF NOT EXISTS knowledge.db_knowledge (
        id SERIAL PRIMARY KEY,
        db_name VARCHAR(50) NOT NULL,
        question TEXT NOT NULL,
        sql TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS knowledge.business_glossary (
        id SERIAL PRIMARY KEY,
        db_name VARCHAR(50) NOT NULL,
        term VARCHAR(200) NOT NULL,
        definition TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )
    """

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.engine = DatabasePoolManager.get_engine(db_name)

        if db_name not in _BaseRepo._table_initialized:
            with _BaseRepo._init_lock:
                if db_name not in _BaseRepo._table_initialized:
                    self._ensure_tables()
                    _BaseRepo._table_initialized.add(db_name)

    def _ensure_tables(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text(self._CREATE_SQL))
                conn.commit()
                print(f"   ✅ 知识库表已就绪")
        except Exception:
            print(f"   ⚠️ 知识库表检查失败")


class SQLKnowledgeRepo(_BaseRepo):
    """SQL 知识库 CRUD（knowledge.db_knowledge）"""

    def list(self) -> List[Dict]:
        query = """
        SELECT id, question, sql, created_at, updated_at
        FROM knowledge.db_knowledge
        WHERE db_name = :db_name
        ORDER BY id
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"db_name": self.db_name})
                rows = result.fetchall()
                return [
                    {
                        'id': row[0],
                        'question': row[1],
                        'sql': row[2],
                        'created_at': str(row[3]) if row[3] else None,
                        'updated_at': str(row[4]) if row[4] else None
                    }
                    for row in rows
                ]
        except Exception as e:
            print(f"获取知识库失败: {e}")
            return []

    def add(self, question: str, sql: str) -> Dict:
        if not question or not sql:
            raise ValueError("问题和SQL不能为空")

        insert_query = """
        INSERT INTO knowledge.db_knowledge (db_name, question, sql, created_at, updated_at)
        VALUES (:db_name, :question, :sql, NOW(), NOW())
        RETURNING id
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(insert_query),
                    {"db_name": self.db_name, "question": question, "sql": sql}
                )
                new_id = result.fetchone()[0]
                conn.commit()
                return {'id': new_id, 'question': question, 'sql': sql}
        except Exception as e:
            raise ValueError(f"添加知识条目失败: {e}")

    def update(self, knowledge_id: int, question: str, sql: str) -> bool:
        update_query = """
        UPDATE knowledge.db_knowledge
        SET question = :question, sql = :sql, updated_at = NOW()
        WHERE id = :id AND db_name = :db_name
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(update_query),
                    {"id": knowledge_id, "db_name": self.db_name, "question": question, "sql": sql}
                )
                conn.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"更新知识条目失败: {e}")
            return False

    def delete(self, knowledge_id: int) -> bool:
        delete_query = """
        DELETE FROM knowledge.db_knowledge
        WHERE id = :id AND db_name = :db_name
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(delete_query),
                    {"id": knowledge_id, "db_name": self.db_name}
                )
                conn.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"删除知识条目失败: {e}")
            return False


class GlossaryRepo(_BaseRepo):
    """业务名词 CRUD（knowledge.business_glossary）"""

    def list(self) -> List[Dict]:
        query = """
        SELECT id, term, definition, created_at, updated_at
        FROM knowledge.business_glossary
        WHERE db_name = :db_name
        ORDER BY id
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"db_name": self.db_name})
                rows = result.fetchall()
                return [
                    {
                        'id': row[0],
                        'term': row[1],
                        'definition': row[2],
                        'created_at': str(row[3]) if row[3] else None,
                        'updated_at': str(row[4]) if row[4] else None
                    }
                    for row in rows
                ]
        except Exception as e:
            print(f"获取业务名词失败: {e}")
            return []

    def add(self, term: str, definition: str) -> Dict:
        if not term or not definition:
            raise ValueError("名词和释义不能为空")
        insert_query = """
        INSERT INTO knowledge.business_glossary (db_name, term, definition, created_at, updated_at)
        VALUES (:db_name, :term, :definition, NOW(), NOW())
        RETURNING id
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(insert_query),
                    {"db_name": self.db_name, "term": term, "definition": definition}
                )
                new_id = result.fetchone()[0]
                conn.commit()
                return {'id': new_id, 'term': term, 'definition': definition}
        except Exception as e:
            raise ValueError(f"添加业务名词失败: {e}")

    def update(self, glossary_id: int, term: str, definition: str) -> bool:
        update_query = """
        UPDATE knowledge.business_glossary
        SET term = :term, definition = :definition, updated_at = NOW()
        WHERE id = :id AND db_name = :db_name
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(update_query),
                    {"id": glossary_id, "db_name": self.db_name, "term": term, "definition": definition}
                )
                conn.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"更新业务名词失败: {e}")
            return False

    def delete(self, glossary_id: int) -> bool:
        delete_query = """
        DELETE FROM knowledge.business_glossary
        WHERE id = :id AND db_name = :db_name
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(delete_query),
                    {"id": glossary_id, "db_name": self.db_name}
                )
                conn.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"删除业务名词失败: {e}")
            return False
