# core/converter.py - 顶层 Text2SQL 转换器 + 预计算入口
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .db_manager import DatabaseManager
from .knowledge import KnowledgeBase
from .llm_client import DouBaoClient
from .repos import SQLKnowledgeRepo, GlossaryRepo
from .utils import SENTENCE_TRANSFORMER_MODEL, monitor_function
from .vector_search import TableSchemaSearcher


class TextToSQLConverter:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.db = DatabaseManager(db_name)
        self.kb = KnowledgeBase(db_name)
        self.sql_repo = SQLKnowledgeRepo(db_name)
        self.glossary_repo = GlossaryRepo(db_name)
        api_key = os.getenv("ARK_API_KEY")
        if not api_key:
            raise ValueError("ARK_API_KEY environment variable is required")
        self.llm = DouBaoClient(api_key=api_key)

    @monitor_function
    def execute_nl_query(
        self,
        nl_query: str,
        selected_table: Optional[str] = None,
        use_vector_search: bool = True,
        top_k_tables: int = 10,
        force_rebuild_vectors: bool = False,
        schema_filter: Optional[str] = None,
    ) -> Tuple[str, pd.DataFrame]:
        print(f"\n📝 查询: {nl_query}")
        if schema_filter:
            print(f"🏷️  Schema 过滤: {schema_filter}")

        if selected_table:
            print(f"🎯 指定表模式: {selected_table}")
            formatted_tables = self.kb.get_table_schema_by_name(selected_table)
            if not formatted_tables or formatted_tables.startswith("未找到表"):
                print(f"   ⚠️ 未找到指定表 {selected_table}，尝试使用所有表")
                formatted_tables = self.kb.get_formatted_schema(schema_filter=schema_filter)
            else:
                print(f"   ✅ 只传递了表: {selected_table}")
        elif use_vector_search:
            print(f"🔍 向量检索模式: Top {top_k_tables}")
            best_tables = TableSchemaSearcher.search(
                self.db_name, nl_query, top_k_tables, self.kb,
                use_holo_index=True, force_rebuild_vectors=force_rebuild_vectors
            )

            if not best_tables:
                formatted_tables = self.kb.get_formatted_schema(schema_filter=schema_filter)
                print(f"   ⚠️ 向量检索无结果，使用所有表")
            else:
                if schema_filter:
                    filter_set = set(s.strip() for s in schema_filter.split(',') if s.strip())
                    best_tables = [t for t in best_tables if t.get('schema') in filter_set]
                selected_names = []
                for t in best_tables:
                    if t.get('schema'):
                        selected_names.append(f"{t['schema']}.{t['table_name']}")
                    else:
                        selected_names.append(t['table_name'])
                if selected_names:
                    formatted_tables = self.kb.get_formatted_schema(selected_names)
                    print(f"   ✅ 传递了 {len(selected_names)} 个相关表")
                else:
                    formatted_tables = self.kb.get_formatted_schema(schema_filter=schema_filter)
                    print(f"   ⚠️ schema 过滤后无结果，使用该 schema 所有表")
        else:
            print(f"📚 全量模式（所有表结构）")
            formatted_tables = self.kb.get_formatted_schema(schema_filter=schema_filter)

        print(f"    ├─ 传递给AI的表结构长度: {len(formatted_tables)} 字符")

        knowledge_json = self.sql_repo.list()
        glossary = self.glossary_repo.list()
        sql = self.llm.generate_text(nl_query, formatted_tables, knowledge_json, glossary)

        if not sql:
            raise ValueError("AI未能生成有效的SQL语句")

        result = self.db.execute_sql(sql)

        return sql, result


def precompute_all_embeddings(db_name: str = None, force_rebuild: bool = False):
    """预计算所有数据库的表结构向量并保存到Hologres"""

    if db_name:
        db_names = [db_name]
    else:
        from config import get_available_databases
        db_configs = get_available_databases()
        db_names = [db_config['id'] for db_config in db_configs]

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
                print(f"   ├─ Hologres中已有 {vectors_count} 个向量，跳过（使用 --force 强制重建）")
                continue

            print(f"   ├─ 加载向量模型...")
            model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

            print(f"   ├─ 生成向量中...")
            batch_size = 50
            all_embeddings = []
            for i in range(0, len(vector_texts), batch_size):
                batch = vector_texts[i:i+batch_size]
                batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                print(f"   ├─ 已处理 {min(i+batch_size, len(vector_texts))}/{len(vector_texts)}")

            embeddings = np.array(all_embeddings)

            kb.save_embeddings_to_holo(table_records, embeddings)

            print(f"   ✅ 完成！已保存 {len(table_records)} 个向量到Hologres")

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
