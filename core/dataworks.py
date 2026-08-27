# core/dataworks.py - DataWorks 节点知识只读检索
import json
import os
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from .db_manager import DatabasePoolManager


DEFAULT_KNOWLEDGE_DB_NAME = os.getenv('AUTH_DB_NAME', os.getenv('APP_AUTH_DB_NAME', 'hologres'))
DEFAULT_PROJECT_ID = os.getenv('DATAWORKS_PROJECT_ID', '')
DEFAULT_TOP_K = 5
CONTENT_LIMIT = 6000
TOKEN_LIMIT = 8


def _normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_int(value: Any) -> Optional[int]:
    text_value = _normalize_text(value)
    if not text_value:
        return None
    try:
        return int(text_value)
    except Exception:
        return None


def _normalize_json_list(value: Any) -> List[Any]:
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


def _compact_text(value: Any, limit: int = CONTENT_LIMIT) -> str:
    text_value = _normalize_text(value)
    if len(text_value) <= limit:
        return text_value
    return text_value[:limit] + '\n...(内容已截断)'


def _tokenize_query(query: str) -> List[str]:
    text_value = _normalize_text(query)
    if not text_value:
        return []

    tokens = [text_value]
    tokens.extend(re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z_][A-Za-z0-9_.$:-]*|\d+', text_value))

    unique_tokens = []
    for token in tokens:
        token = token.strip().strip('`"\'')
        if token and token not in unique_tokens:
            unique_tokens.append(token)
    return unique_tokens[:TOKEN_LIMIT]


def _format_list(values: Any) -> str:
    items = []
    for value in _normalize_json_list(values):
        if isinstance(value, dict):
            name = _normalize_text(
                value.get('node_name')
                or value.get('file_name')
                or value.get('table_name')
                or value.get('node_id')
                or value.get('id')
            )
        else:
            name = _normalize_text(value)
        if name:
            items.append(name)
    return ', '.join(dict.fromkeys(items))


class DataWorksKnowledgeStore:
    """读取 DataWorks/MaxCompute 侧已落好的节点代码知识表。"""

    _schema_ready: set[str] = set()

    def __init__(
        self,
        db_name: Optional[str] = None,
        project_id: Optional[int] = None,
        project_identifier: Optional[str] = None,
        ensure_schema: bool = False,
        **_: Any,
    ):
        self.db_name = db_name or DEFAULT_KNOWLEDGE_DB_NAME
        self.project_id = project_id if project_id is not None else _normalize_int(DEFAULT_PROJECT_ID)
        self.project_identifier = project_identifier or ''
        self.ensure_schema = ensure_schema
        self.engine = None
        try:
            self.engine = DatabasePoolManager.get_engine(self.db_name)
        except Exception as exc:
            print(f"   ⚠️ DataWorks 知识表连接不可用，跳过召回: {exc}")

    def _schema_is_ready(self) -> bool:
        if not self.engine:
            return False
        if self.db_name in self._schema_ready:
            return True

        try:
            with self.engine.connect() as conn:
                row = conn.execute(text("""
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'knowledge'
                      AND table_name = 'dataworks_node_knowledge'
                    LIMIT 1
                """)).fetchone()
        except Exception as exc:
            print(f"   ⚠️ DataWorks 知识表不可用，跳过召回: {exc}")
            return False

        if row is not None:
            self._schema_ready.add(self.db_name)
            return True
        return False

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """按关键词从 knowledge.dataworks_node_knowledge 召回节点代码。

        provider 参数保留是为了兼容旧调用方；这里不再做向量检索。
        """
        del provider
        tokens = _tokenize_query(query)
        if not tokens or not self._schema_is_ready():
            return []

        try:
            limit = max(1, min(20, int(top_k or DEFAULT_TOP_K)))
        except Exception:
            limit = DEFAULT_TOP_K

        params: Dict[str, Any] = {'limit': limit}
        search_groups = []
        score_parts = []
        for index, token in enumerate(tokens):
            key = f'pattern_{index}'
            params[key] = f'%{token}%'
            field_match = f"""
                COALESCE(node_name, '') ILIKE :{key}
                OR COALESCE(file_name, '') ILIKE :{key}
                OR COALESCE(absolute_folder_path, '') ILIKE :{key}
                OR COALESCE(file_folder_path, '') ILIKE :{key}
                OR COALESCE(file_description, '') ILIKE :{key}
                OR COALESCE(content, '') ILIKE :{key}
                OR COALESCE(upstream_tables::text, '') ILIKE :{key}
                OR COALESCE(output_tables::text, '') ILIKE :{key}
                OR COALESCE(upstream_nodes::text, '') ILIKE :{key}
            """
            search_groups.append(f'({field_match})')
            score_parts.append(f"""
                CASE
                    WHEN COALESCE(node_name, '') ILIKE :{key}
                      OR COALESCE(file_name, '') ILIKE :{key}
                      OR COALESCE(output_tables::text, '') ILIKE :{key}
                    THEN 3
                    WHEN COALESCE(absolute_folder_path, '') ILIKE :{key}
                      OR COALESCE(file_folder_path, '') ILIKE :{key}
                      OR COALESCE(upstream_tables::text, '') ILIKE :{key}
                    THEN 2
                    WHEN COALESCE(file_description, '') ILIKE :{key}
                      OR COALESCE(content, '') ILIKE :{key}
                      OR COALESCE(upstream_nodes::text, '') ILIKE :{key}
                    THEN 1
                    ELSE 0
                END
            """)

        db_names = [self.db_name]
        if DEFAULT_KNOWLEDGE_DB_NAME and DEFAULT_KNOWLEDGE_DB_NAME not in db_names:
            db_names.append(DEFAULT_KNOWLEDGE_DB_NAME)
        db_clauses = []
        for index, name in enumerate(db_names):
            key = f'db_name_{index}'
            params[key] = name
            db_clauses.append(f'db_name = :{key}')

        where_clauses = [
            f"({' OR '.join(db_clauses)})",
            "COALESCE(is_active, TRUE) IS TRUE",
            f"({' OR '.join(search_groups)})",
        ]
        if self.project_id is not None:
            params['project_id'] = self.project_id
            where_clauses.append('project_id = :project_id')

        sql = text(f"""
            SELECT
                id,
                db_name,
                project_id,
                project_identifier,
                workspace_region,
                node_key,
                node_id,
                file_id,
                node_name,
                file_name,
                file_folder_path,
                absolute_folder_path,
                file_type,
                use_type,
                connection_name,
                owner,
                last_edit_user,
                commit_status,
                auto_parsing,
                is_maxcompute,
                current_version,
                file_description,
                source_modified_at,
                content,
                input_list,
                output_list,
                dependent_node_ids,
                upstream_nodes,
                upstream_tables,
                output_tables,
                node_configuration,
                file_payload,
                text_hash,
                source_hash,
                last_seen_at,
                created_at,
                updated_at,
                ({' + '.join(score_parts)}) AS keyword_score
            FROM knowledge.dataworks_node_knowledge
            WHERE {' AND '.join(where_clauses)}
            ORDER BY keyword_score DESC,
                     updated_at DESC NULLS LAST,
                     source_modified_at DESC NULLS LAST,
                     id DESC
            LIMIT :limit
        """)

        try:
            with self.engine.connect() as conn:
                rows = conn.execute(sql, params).mappings().all()
        except Exception as exc:
            print(f"   ⚠️ DataWorks 知识召回失败，跳过: {exc}")
            return []

        results = []
        for rank, row in enumerate(rows, start=1):
            item = dict(row)
            for field in (
                'input_list',
                'output_list',
                'dependent_node_ids',
                'upstream_nodes',
                'upstream_tables',
                'output_tables',
            ):
                item[field] = _normalize_json_list(item.get(field))
            item['content'] = _compact_text(item.get('content'))
            item['_rank'] = rank
            item['_score'] = float(item.get('keyword_score') or 0)
            results.append(item)
        return results

    @staticmethod
    def format_results_for_prompt(results: List[Dict[str, Any]], limit: int = DEFAULT_TOP_K) -> str:
        if not results:
            return '无可用 DataWorks 节点知识'

        blocks = []
        for item in results[:limit]:
            name = _normalize_text(item.get('node_name') or item.get('file_name') or item.get('node_key')) or '未命名节点'
            path = _normalize_text(item.get('absolute_folder_path') or item.get('file_folder_path'))
            content = _compact_text(item.get('content'), limit=1200)
            lines = [f"节点: {name}"]
            if path:
                lines.append(f"路径: {path}")
            outputs = _format_list(item.get('output_tables'))
            upstream_nodes = _format_list(item.get('upstream_nodes'))
            upstream_tables = _format_list(item.get('upstream_tables'))
            if outputs:
                lines.append(f"输出表: {outputs}")
            if upstream_nodes:
                lines.append(f"上游节点: {upstream_nodes}")
            if upstream_tables:
                lines.append(f"上游表: {upstream_tables}")
            if content:
                lines.append(f"代码:\n{content}")
            blocks.append('\n'.join(lines))

        return '\n\n'.join(blocks)
