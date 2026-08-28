# core/converter.py - 顶层 Text2SQL 转换器 + 预计算入口
import os
import re
import time
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .cancellation import CancellationToken, CancelledError
from .db_manager import DatabaseManager
from .embedding_client import iter_embedding_models
from .book_code import BookKnowledgeStore
from .dataworks import DataWorksKnowledgeStore
from .knowledge import KnowledgeBase
from .llm_client import DouBaoClient
from .query_log import insert_query_log
from .repos import SQLKnowledgeRepo, GlossaryRepo
from .utils import monitor_function
from .vector_search import TableSchemaSearcher


class TextToSQLConverter:
    _DATAWORKS_CONTEXT_MODES = {'bi', 'xb_bi'}
    _BOOK_CONTEXT_MODES = {'bi', 'xb_bi', 'book'}
    _ALL_CONTEXT_MODES = _DATAWORKS_CONTEXT_MODES | {'book'}

    def __init__(self, db_name: str, current_user: Optional[dict] = None):
        self.db_name = db_name
        self.current_user = current_user
        self.db = DatabaseManager(db_name)
        self.kb = KnowledgeBase(db_name)
        self.dataworks_store = DataWorksKnowledgeStore(db_name, ensure_schema=False)
        self.book_store = BookKnowledgeStore(db_name, ensure_schema=False)
        self.sql_repo = SQLKnowledgeRepo(db_name, current_user=current_user)
        self.glossary_repo = GlossaryRepo(db_name, current_user=current_user)
        api_key = os.getenv("ARK_API_KEY")
        if not api_key:
            raise ValueError("ARK_API_KEY environment variable is required")
        self.llm = DouBaoClient(api_key=api_key)

    @classmethod
    def _split_mode_values(cls, *values: Optional[str]) -> list[str]:
        modes = []
        for value in values:
            for part in re.split(r'[,，;；\s]+', str(value or '')):
                mode = part.strip().lower().replace('-', '_')
                if mode:
                    modes.append(mode)
        return modes

    @classmethod
    def _resolve_context_mode(cls, explicit_mode: Optional[str],
                              schema_filter: Optional[str]) -> Optional[str]:
        for mode in cls._split_mode_values(explicit_mode):
            if mode in cls._ALL_CONTEXT_MODES:
                return mode

        schema_modes = cls._split_mode_values(schema_filter)
        if 'book' in schema_modes and not any(mode in cls._DATAWORKS_CONTEXT_MODES for mode in schema_modes):
            return 'book'
        for mode in schema_modes:
            if mode in cls._DATAWORKS_CONTEXT_MODES:
                return mode
        if 'book' in schema_modes:
            return 'book'
        return None

    @classmethod
    def _schema_filter_for_table_search(cls, schema_filter: Optional[str]) -> Optional[str]:
        schemas = [
            part.strip()
            for part in re.split(r'[,，;；]+', str(schema_filter or ''))
            if part.strip()
        ]
        schemas = [
            schema
            for schema in schemas
            if schema.lower().replace('-', '_') != 'book'
        ]
        return ','.join(schemas) if schemas else None

    @classmethod
    def _iter_text_values(cls, value: Any) -> Iterable[str]:
        if value is None:
            return
        if isinstance(value, str):
            if value.strip():
                yield value
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str) and key.strip():
                    yield key
                yield from cls._iter_text_values(child)
            return
        if isinstance(value, (list, tuple, set)):
            for child in value:
                yield from cls._iter_text_values(child)
            return
        text_value = str(value).strip()
        if text_value:
            yield text_value

    @classmethod
    def _extract_table_hints_from_book_knowledge(cls, book_knowledge: list[dict]) -> list[str]:
        if not book_knowledge:
            return []

        sql_ref_pattern = re.compile(
            r'(?i)\b(?:from|join|into|update|table|overwrite\s+table|insert\s+(?:overwrite\s+)?(?:into\s+)?(?:table\s+)?)'
            r'\s+[`"]?([A-Za-z_][A-Za-z0-9_]*)[`"]?\s*\.\s*[`"]?([A-Za-z_][A-Za-z0-9_]*)[`"]?'
        )
        known_schema_pattern = re.compile(
            r'(?i)(?<![A-Za-z0-9_])(bi|xb_bi)\s*\.\s*[`"]?([A-Za-z_][A-Za-z0-9_]*)[`"]?'
        )

        hints = []
        seen = set()

        def add(schema: str, table: str):
            schema = str(schema or '').strip('`" ').lower()
            table = str(table or '').strip('`" ')
            if not schema or not table:
                return
            full_name = f'{schema}.{table}'
            key = full_name.lower()
            if key not in seen:
                seen.add(key)
                hints.append(full_name)

        fields = (
            'file_path',
            'qualified_name',
            'symbol_name',
            'signature',
            'docstring',
            'leading_comments',
            'code_text',
            'context_text',
            'vector_text',
            'references_json',
            'context_json',
        )
        for item in book_knowledge[:8]:
            for field in fields:
                for text_value in cls._iter_text_values(item.get(field)):
                    for match in sql_ref_pattern.finditer(text_value):
                        add(match.group(1), match.group(2))
                    for match in known_schema_pattern.finditer(text_value):
                        add(match.group(1), match.group(2))

        return hints

    @classmethod
    def _build_book_informed_table_query(cls, nl_query: str, book_knowledge: list[dict],
                                         table_hints: list[str]) -> str:
        parts = [nl_query]
        parts.extend(table_hints[:12])
        for item in book_knowledge[:3]:
            for field in ('qualified_name', 'symbol_name', 'file_path', 'docstring', 'leading_comments'):
                value = str(item.get(field) or '').strip()
                if value:
                    parts.append(value[:300])
        query = '\n'.join(part for part in parts if part)
        return query[:4000]

    @staticmethod
    def _table_result_key(table: dict) -> str:
        schema = str(table.get('schema') or '').strip().lower()
        table_name = str(table.get('table_name') or '').strip().lower()
        if not table_name:
            return ''
        return f'{schema}.{table_name}' if schema else table_name

    @classmethod
    def _merge_table_results(cls, primary: list[dict], fallback: list[dict],
                             limit: int) -> list[dict]:
        try:
            limit_value = max(1, int(limit))
        except Exception:
            limit_value = 10
        merged: dict[str, dict] = {}
        scores: dict[str, float] = {}
        source_order: dict[str, int] = {}

        for source_index, rows in enumerate((primary or [], fallback or [])):
            if not rows:
                continue
            weight = 1.0 if source_index == 0 else 0.85
            total = len(rows)
            for rank, row in enumerate(rows, 1):
                key = cls._table_result_key(row)
                if not key:
                    continue
                if key not in merged:
                    merged[key] = dict(row)
                    source_order[key] = source_index
                scores[key] = scores.get(key, 0.0) + weight * ((total - rank + 1) / total)

        sorted_keys = sorted(
            merged,
            key=lambda key: (-scores.get(key, 0.0), source_order.get(key, 99), key),
        )
        results = []
        for rank, key in enumerate(sorted_keys[:limit_value], 1):
            item = merged[key]
            item['_rank'] = rank
            item['_combined_score'] = scores.get(key, 0.0)
            results.append(item)
        return results

    @staticmethod
    def _table_names_from_results(tables: list[dict]) -> list[str]:
        names = []
        for table in tables or []:
            table_name = table.get('table_name')
            schema_name = table.get('schema')
            if table_name:
                names.append(f"{schema_name}.{table_name}" if schema_name else table_name)
        return list(dict.fromkeys(names))

    @monitor_function
    def execute_nl_query(
        self,
        nl_query: str,
        selected_table: Optional[str] = None,
        use_vector_search: bool = True,
        top_k_tables: int = 10,
        force_rebuild_vectors: bool = False,
        schema_filter: Optional[str] = None,
        cancel_token: Optional[CancellationToken] = None,
        embedding_provider: Optional[str] = None,
        request_id: Optional[str] = None,
        query_context_mode: Optional[str] = None,
    ) -> Tuple[str, pd.DataFrame]:
        print(f"\n📝 查询: {nl_query}")
        if schema_filter:
            print(f"🏷️  Schema 过滤: {schema_filter}")
        context_mode = self._resolve_context_mode(query_context_mode, schema_filter)
        schema_filter_for_tables = self._schema_filter_for_table_search(schema_filter)
        print(f"🧭 查询上下文模式: {context_mode or '默认'}")

        t_start = time.time()
        log_data = {
            'db_name': self.db_name,
            'nl_query': nl_query,
            'schema_filter': schema_filter,
            'selected_table': selected_table,
            'top_k': top_k_tables if use_vector_search and not selected_table else None,
            'search_mode': 'selected_table' if selected_table else ('vector' if use_vector_search else 'all'),
            'matched_tables': None,
            'generated_sql': None,
            'execute_status': 'failed',
            'error_message': None,
            'result_rows': None,
            'search_duration_ms': None,
            'llm_duration_ms': None,
            'sql_exec_duration_ms': None,
            'total_duration_ms': None,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'llm_calls': 0,
            'request_id': request_id,
            'action_type': 'nl_query',
        }

        sql = None
        result = None

        try:
            best_tables = None  # 向量检索结果，给候选 SQL 评分用
            book_knowledge = []
            book_table_hints = []
            selected_table_names = []

            def _check_cancel():
                if cancel_token is not None:
                    cancel_token.raise_if_cancelled()

            _check_cancel()

            if context_mode in self._BOOK_CONTEXT_MODES:
                mode_label = 'book' if context_mode == 'book' else f'{context_mode}/Git 优先'
                print(f"📚 {mode_label} 模式: 先检索 Git 本地代码知识，再用代码线索检索表结构")
                book_knowledge = self.book_store.search(
                    nl_query,
                    top_k=5,
                    provider=embedding_provider,
                    search_mode='vector',
                )
                book_table_hints = self._extract_table_hints_from_book_knowledge(book_knowledge)
                if book_table_hints:
                    print(f"   ├─ Git 代码中识别到表线索: {', '.join(book_table_hints[:8])}")
                _check_cancel()

            if selected_table:
                print(f"🎯 指定表模式: {selected_table}")
                formatted_tables = self.kb.get_table_schema_by_name(selected_table)
                if not formatted_tables or formatted_tables.startswith("未找到表"):
                    print(f"   ⚠️ 未找到指定表 {selected_table}，尝试使用所有表")
                    formatted_tables = self.kb.get_formatted_schema(schema_filter=schema_filter_for_tables)
                else:
                    print(f"   ✅ 只传递了表: {selected_table}")
                    selected_table_names = [selected_table]
            elif use_vector_search:
                print(f"🔍 向量检索模式: Top {top_k_tables}")
                t_search_start = time.time()
                table_search_query = nl_query
                if context_mode in self._BOOK_CONTEXT_MODES:
                    table_search_query = self._build_book_informed_table_query(
                        nl_query,
                        book_knowledge,
                        book_table_hints,
                    )
                primary_tables = TableSchemaSearcher.search(
                    self.db_name, table_search_query, top_k_tables, self.kb,
                    use_holo_index=True, force_rebuild_vectors=force_rebuild_vectors,
                    schema_filter=schema_filter_for_tables,
                    embedding_provider=embedding_provider,
                )
                if context_mode in self._DATAWORKS_CONTEXT_MODES and table_search_query != nl_query:
                    print("   ├─ 保留用户原问题表检索兜底，并与 Git 增强结果合并")
                    fallback_tables = TableSchemaSearcher.search(
                        self.db_name, nl_query, top_k_tables, self.kb,
                        use_holo_index=True, force_rebuild_vectors=False,
                        schema_filter=schema_filter_for_tables,
                        embedding_provider=embedding_provider,
                    )
                    best_tables = self._merge_table_results(primary_tables, fallback_tables, top_k_tables)
                else:
                    best_tables = primary_tables
                log_data['search_duration_ms'] = (time.time() - t_search_start) * 1000
                _check_cancel()

                if not best_tables:
                    if context_mode in self._BOOK_CONTEXT_MODES and book_table_hints:
                        formatted_tables = self.kb.get_formatted_schema(book_table_hints)
                        if formatted_tables.startswith("未找到指定的表"):
                            formatted_tables = self.kb.get_formatted_schema(schema_filter=schema_filter_for_tables)
                    else:
                        formatted_tables = self.kb.get_formatted_schema(schema_filter=schema_filter_for_tables)
                    print(f"   ⚠️ 向量检索无结果，使用所有表")
                else:
                    selected_names = []
                    if context_mode in self._BOOK_CONTEXT_MODES:
                        selected_names.extend(book_table_hints[:5])
                    selected_names.extend(self._table_names_from_results(best_tables))
                    selected_names = list(dict.fromkeys(selected_names))
                    selected_table_names = selected_names
                    log_data['matched_tables'] = selected_names
                    formatted_tables = self.kb.get_formatted_schema(selected_names)
                    print(f"   ✅ 传递了 {len(selected_names)} 个相关表")
            else:
                print(f"📚 全量模式（所有表结构）")
                formatted_tables = self.kb.get_formatted_schema(schema_filter=schema_filter_for_tables)

            print(f"    ├─ 传递给AI的表结构长度: {len(formatted_tables)} 字符")
            _check_cancel()

            knowledge_json = self.sql_repo.list()
            glossary = self.glossary_repo.list()
            if context_mode in self._DATAWORKS_CONTEXT_MODES or context_mode is None:
                dataworks_query = nl_query
                if context_mode in self._DATAWORKS_CONTEXT_MODES:
                    dataworks_query = self._build_book_informed_table_query(
                        nl_query,
                        book_knowledge,
                        selected_table_names or book_table_hints,
                    )
                dataworks_knowledge = self.dataworks_store.search(dataworks_query, top_k=5, provider=embedding_provider)
            else:
                dataworks_knowledge = []

            if not book_knowledge and (context_mode in self._BOOK_CONTEXT_MODES or context_mode is None):
                code_search_mode = 'selected_table' if selected_table else ('vector' if use_vector_search else 'all')
                book_table_hints = []
                if selected_table:
                    book_table_hints.append(selected_table)
                elif best_tables:
                    for table in best_tables:
                        table_name = table.get('table_name')
                        schema_name_hint = table.get('schema')
                        if table_name:
                            book_table_hints.append(f"{schema_name_hint}.{table_name}" if schema_name_hint else table_name)
                book_knowledge = self.book_store.search(
                    nl_query,
                    top_k=5,
                    provider=embedding_provider,
                    search_mode=code_search_mode,
                    table_hints=book_table_hints,
                )

            t_llm_start = time.time()
            sql = self.llm.generate_text(nl_query, formatted_tables, knowledge_json, glossary,
                                         vector_results=best_tables,
                                         dataworks_knowledge=dataworks_knowledge,
                                         book_knowledge=book_knowledge,
                                         cancel_token=cancel_token)
            log_data['llm_duration_ms'] = (time.time() - t_llm_start) * 1000
            log_data['generated_sql'] = sql

            # 记录 token 用量
            usage = getattr(self.llm, 'last_usage', {}) or {}
            log_data['prompt_tokens'] = usage.get('prompt_tokens', 0)
            log_data['completion_tokens'] = usage.get('completion_tokens', 0)
            log_data['total_tokens'] = usage.get('total_tokens', 0)
            log_data['llm_calls'] = usage.get('calls', 0)

            if not sql:
                raise ValueError("AI未能生成有效的SQL语句")

            _check_cancel()
            t_exec_start = time.time()
            result = self.db.execute_sql(sql, cancel_token=cancel_token)
            log_data['sql_exec_duration_ms'] = (time.time() - t_exec_start) * 1000
            log_data['result_rows'] = len(result) if result is not None else 0
            log_data['execute_status'] = 'success'

            return sql, result

        except CancelledError as e:
            log_data['error_message'] = '用户已取消'
            log_data['execute_status'] = 'cancelled'
            raise

        except Exception as e:
            log_data['error_message'] = str(e)[:1000]
            log_data['execute_status'] = 'failed'
            raise

        finally:
            log_data['total_duration_ms'] = (time.time() - t_start) * 1000
            insert_query_log(self.db_name, log_data)


def precompute_all_embeddings(db_name: str = None, force_rebuild: bool = False):
    """预计算所有数据库的表结构向量并保存到Hologres"""

    if db_name:
        db_names = [db_name]
    else:
        from config import get_available_databases, EXCLUDED_DATABASES
        db_configs = get_available_databases()
        db_names = [db['id'] for db in db_configs if db['id'] not in EXCLUDED_DATABASES]

    for name in db_names:
        print(f"\n{'='*60}")
        print(f"处理数据库: {name}")
        print(f"{'='*60}")

        try:
            kb = KnowledgeBase(name)

            table_records, vector_texts = kb.get_vector_texts()

            if not table_records:
                print(f"   ⚠️ 没有找到任何表")
                continue

            print(f"   ├─ 找到 {len(table_records)} 个表")

            vectors_count = kb.get_holo_vectors_count()
            if vectors_count > 0 and not force_rebuild:
                # 增量同步两个模型列
                print(f"   ├─ Hologres 中已有 {vectors_count} 个向量，走增量同步（使用 --force 可强制全量重建）")
                for provider, model in iter_embedding_models():
                    print(f"   ├─ 加载向量模型 ({provider})...")
                    kb.save_embeddings_incrementally(model, table_records, vector_texts)
                continue

            print(f"   ├─ 全量生成向量中（两个模型）...")
            for provider, model in iter_embedding_models():
                print(f"   ├─ 加载向量模型 ({provider})...")

                batch_size = 50
                all_embeddings = []
                for i in range(0, len(vector_texts), batch_size):
                    batch = vector_texts[i:i+batch_size]
                    batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
                    all_embeddings.extend(batch_embeddings)
                    print(f"   ├─ 已处理 {min(i+batch_size, len(vector_texts))}/{len(vector_texts)} ({provider})")

                embeddings = np.array(all_embeddings)
                embedding_col = KnowledgeBase._embedding_col(provider)
                kb.save_embeddings_to_holo(table_records, embeddings, embedding_col=embedding_col)

            print(f"   ✅ 完成！已保存 {len(table_records)} 个向量到Hologres")

            # 同时重建知识库和业务名词的向量
            print(f"   ├─ 重建知识库/名词向量...")
            kb.rebuild_knowledge_vectors()

        except Exception as e:
            print(f"   ❌ 处理数据库 {name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "precompute":
        if len(sys.argv) > 2:
            db_name_arg = sys.argv[2]
        else:
            db_name_arg = None

        force = len(sys.argv) > 3 and sys.argv[3] == "--force"

        precompute_all_embeddings(db_name_arg, force)
    else:
        print("用法:")
        print("  python -m core.converter precompute              # 预计算所有数据库向量")
        print("  python -m core.converter precompute your_db      # 预计算指定数据库向量")
        print("  python -m core.converter precompute your_db --force  # 强制重建")
