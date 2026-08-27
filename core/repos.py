# core/repos.py - 知识库（SQL 知识 + 业务名词）的 CRUD 仓储类
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text

from .auth import user_can_view_all
from .db_manager import DatabasePoolManager


class _BaseRepo:
    """知识库 / 名词仓库基类"""

    def __init__(self, db_name: str, current_user: Optional[Dict] = None):
        self.db_name = db_name
        self.current_user = current_user
        self.engine = DatabasePoolManager.get_engine(db_name)

    def _visibility_scope(self) -> str:
        return 'all' if user_can_view_all(self.current_user) else 'self'

    def _audit_params(self) -> Dict:
        user = self.current_user or {}
        return {
            'created_by_admin_id': user.get('admin_id'),
            'created_by_user_name': user.get('user_name'),
            'updated_by_admin_id': user.get('admin_id'),
            'updated_by_user_name': user.get('user_name'),
            'visibility_scope': self._visibility_scope(),
        }

    def _permission_clause(self) -> Tuple[str, Dict]:
        if not self.current_user or user_can_view_all(self.current_user):
            return '', {}
        return (
            " AND (created_by_admin_id = :current_admin_id "
            "OR visibility_scope = 'all' OR created_by_admin_id IS NULL)",
            {'current_admin_id': self.current_user.get('admin_id')},
        )

    def _write_permission_clause(self) -> Tuple[str, Dict]:
        if not self.current_user or user_can_view_all(self.current_user):
            return '', {}
        return (
            " AND created_by_admin_id = :current_admin_id",
            {'current_admin_id': self.current_user.get('admin_id')},
        )


class SQLKnowledgeRepo(_BaseRepo):
    """SQL 知识库 CRUD（knowledge.db_knowledge）"""

    def list(self) -> List[Dict]:
        permission_sql, permission_params = self._permission_clause()
        query = f"""
        SELECT id, question, sql, created_at, updated_at,
               created_by_admin_id, created_by_user_name,
               updated_by_admin_id, updated_by_user_name, visibility_scope
        FROM knowledge.db_knowledge
        WHERE db_name = :db_name
        {permission_sql}
        ORDER BY id
        """
        try:
            with self.engine.connect() as conn:
                params = {"db_name": self.db_name, **permission_params}
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                return [
                    {
                        'id': row[0],
                        'question': row[1],
                        'sql': row[2],
                        'created_at': str(row[3]) if row[3] else None,
                        'updated_at': str(row[4]) if row[4] else None,
                        'created_by_admin_id': row[5],
                        'created_by_user_name': row[6],
                        'updated_by_admin_id': row[7],
                        'updated_by_user_name': row[8],
                        'visibility_scope': row[9],
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
        INSERT INTO knowledge.db_knowledge (
            id, db_name, question, sql,
            created_by_admin_id, created_by_user_name,
            updated_by_admin_id, updated_by_user_name, visibility_scope,
            created_at, updated_at
        )
        VALUES (
            COALESCE((SELECT MAX(id) FROM knowledge.db_knowledge), 0) + 1,
            :db_name, :question, :sql,
            :created_by_admin_id, :created_by_user_name,
            :updated_by_admin_id, :updated_by_user_name, :visibility_scope,
            NOW(), NOW()
        )
        RETURNING id
        """
        try:
            with self.engine.connect() as conn:
                params = {
                    "db_name": self.db_name,
                    "question": question,
                    "sql": sql,
                    **self._audit_params(),
                }
                result = conn.execute(
                    text(insert_query),
                    params,
                )
                new_id = result.fetchone()[0]
                conn.commit()
                return {'id': new_id, 'question': question, 'sql': sql}
        except Exception as e:
            raise ValueError(f"添加知识条目失败: {e}")

    def update(self, knowledge_id: int, question: str, sql: str) -> bool:
        permission_sql, permission_params = self._write_permission_clause()
        update_query = f"""
        UPDATE knowledge.db_knowledge
        SET question = :question,
            sql = :sql,
            updated_by_admin_id = :updated_by_admin_id,
            updated_by_user_name = :updated_by_user_name,
            updated_at = NOW()
        WHERE id = :id AND db_name = :db_name
        {permission_sql}
        """
        try:
            with self.engine.connect() as conn:
                audit = self._audit_params()
                result = conn.execute(
                    text(update_query),
                    {
                        "id": knowledge_id,
                        "db_name": self.db_name,
                        "question": question,
                        "sql": sql,
                        "updated_by_admin_id": audit.get("updated_by_admin_id"),
                        "updated_by_user_name": audit.get("updated_by_user_name"),
                        **permission_params,
                    },
                )
                conn.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"更新知识条目失败: {e}")
            return False

    def delete(self, knowledge_id: int) -> bool:
        permission_sql, permission_params = self._write_permission_clause()
        delete_query = f"""
        DELETE FROM knowledge.db_knowledge
        WHERE id = :id AND db_name = :db_name
        {permission_sql}
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(delete_query),
                    {"id": knowledge_id, "db_name": self.db_name, **permission_params},
                )
                conn.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"删除知识条目失败: {e}")
            return False


class GlossaryRepo(_BaseRepo):
    """业务名词 CRUD（knowledge.business_glossary）"""

    def list(self) -> List[Dict]:
        permission_sql, permission_params = self._permission_clause()
        query = f"""
        SELECT id, term, definition, created_at, updated_at,
               created_by_admin_id, created_by_user_name,
               updated_by_admin_id, updated_by_user_name, visibility_scope
        FROM knowledge.business_glossary
        WHERE db_name = :db_name
        {permission_sql}
        ORDER BY id
        """
        try:
            with self.engine.connect() as conn:
                params = {"db_name": self.db_name, **permission_params}
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                return [
                    {
                        'id': row[0],
                        'term': row[1],
                        'definition': row[2],
                        'created_at': str(row[3]) if row[3] else None,
                        'updated_at': str(row[4]) if row[4] else None,
                        'created_by_admin_id': row[5],
                        'created_by_user_name': row[6],
                        'updated_by_admin_id': row[7],
                        'updated_by_user_name': row[8],
                        'visibility_scope': row[9],
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
        INSERT INTO knowledge.business_glossary (
            db_name, term, definition,
            created_by_admin_id, created_by_user_name,
            updated_by_admin_id, updated_by_user_name, visibility_scope,
            created_at, updated_at
        )
        VALUES (
            :db_name, :term, :definition,
            :created_by_admin_id, :created_by_user_name,
            :updated_by_admin_id, :updated_by_user_name, :visibility_scope,
            NOW(), NOW()
        )
        RETURNING id
        """
        try:
            with self.engine.connect() as conn:
                params = {
                    "db_name": self.db_name,
                    "term": term,
                    "definition": definition,
                    **self._audit_params(),
                }
                result = conn.execute(
                    text(insert_query),
                    params,
                )
                new_id = result.fetchone()[0]
                conn.commit()
                return {'id': new_id, 'term': term, 'definition': definition}
        except Exception as e:
            raise ValueError(f"添加业务名词失败: {e}")

    def update(self, glossary_id: int, term: str, definition: str) -> bool:
        permission_sql, permission_params = self._write_permission_clause()
        update_query = f"""
        UPDATE knowledge.business_glossary
        SET term = :term,
            definition = :definition,
            updated_by_admin_id = :updated_by_admin_id,
            updated_by_user_name = :updated_by_user_name,
            updated_at = NOW()
        WHERE id = :id AND db_name = :db_name
        {permission_sql}
        """
        try:
            with self.engine.connect() as conn:
                audit = self._audit_params()
                result = conn.execute(
                    text(update_query),
                    {
                        "id": glossary_id,
                        "db_name": self.db_name,
                        "term": term,
                        "definition": definition,
                        "updated_by_admin_id": audit.get("updated_by_admin_id"),
                        "updated_by_user_name": audit.get("updated_by_user_name"),
                        **permission_params,
                    },
                )
                conn.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"更新业务名词失败: {e}")
            return False

    def delete(self, glossary_id: int) -> bool:
        permission_sql, permission_params = self._write_permission_clause()
        delete_query = f"""
        DELETE FROM knowledge.business_glossary
        WHERE id = :id AND db_name = :db_name
        {permission_sql}
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(delete_query),
                    {"id": glossary_id, "db_name": self.db_name, **permission_params},
                )
                conn.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"删除业务名词失败: {e}")
            return False
