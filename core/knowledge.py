# core/knowledge.py - KnowledgeBase：表结构查询、向量存取、格式化
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text

from config import get_database_config
from .db_manager import DatabasePoolManager
from .utils import _schema_cache, _vectors_cache, EMBEDDING_DIM, monitor_function


class KnowledgeBase:
    """负责表结构查询、向量存取、格式化（不再包含知识库/名词 CRUD，CRUD 见 repos.py）"""

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.engine = DatabasePoolManager.get_engine(db_name)

        self.cache_dir = Path("./cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dim = EMBEDDING_DIM

    @staticmethod
    def _embedding_col(provider: str) -> str:
        return 'doubao_embedding' if provider == 'api' else 'local_embedding'

    # ========== 向量操作方法 ==========

    def save_embeddings_to_holo(self, table_records: List[Dict], embeddings: np.ndarray,
                                 embedding_col: str = "local_embedding"):
        """批量保存向量到 Hologres（UPDATE 存在的行，INSERT 不存在的行）"""
        if not table_records:
            return True

        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                # Step 1: 查询当前已有的 (db_name, schema_name, table_name) → id 映射
                existing_rows = conn.execute(text("""
                    SELECT COALESCE(schema_name,''), table_name, id
                    FROM knowledge.table_embeddings
                    WHERE db_name = :db_name
                """), {"db_name": self.db_name}).fetchall()
                existing_map = {(r[0], r[1]): r[2] for r in existing_rows}

                # Step 2: 构造批量参数
                batch_update, batch_insert = [], []
                for record, embedding in zip(table_records, embeddings):
                    vector_text = record.get('_vector_text', '')
                    text_hash = hashlib.md5(vector_text.encode()).hexdigest()
                    schema_key = record.get('schema') or ''
                    tbl_key = record['table_name']
                    key = (schema_key, tbl_key)
                    embedding_list = embedding.tolist()
                    column_info_json = json.dumps(record.get('_columns_json', []))
                    table_comment = record.get('table_comment', '')

                    p = {
                        "id": existing_map.get(key),
                        "db_name": self.db_name,
                        "schema_name": record.get('schema'),
                        "table_name": tbl_key,
                        "table_comment": table_comment,
                        "column_info": column_info_json,
                        "vector_text": vector_text,
                        "embedding": embedding_list,
                        "text_hash": text_hash,
                        "schema_hash": text_hash,
                    }

                    if p["id"] is not None:
                        batch_update.append(p)
                    else:
                        batch_insert.append(p)

                # Step 3: 批量 UPDATE 已有行
                if batch_update:
                    conn.execute(text(f"""
                        UPDATE knowledge.table_embeddings
                        SET {embedding_col} = :embedding,
                            text_hash = :text_hash,
                            schema_hash = :schema_hash,
                            vector_text = :vector_text,
                            table_comment = :table_comment,
                            column_info = CAST(:column_info AS jsonb),
                            updated_at = NOW()
                        WHERE id = :id
                    """), batch_update)

                # Step 4: 批量 INSERT 新行
                if batch_insert:
                    conn.execute(text(f"""
                        INSERT INTO knowledge.table_embeddings
                            (db_name, schema_name, table_name, table_comment, column_info,
                             vector_text, {embedding_col}, text_hash, schema_hash, updated_at)
                        VALUES
                            (:db_name, :schema_name, :table_name, :table_comment, CAST(:column_info AS jsonb),
                             :vector_text, :embedding, :text_hash, :schema_hash, NOW())
                    """), batch_insert)

                trans.commit()
                _vectors_cache.invalidate(f"vectors:{self.db_name}")

                self._update_schema_fingerprint(conn, table_records)

                print(f"   ✅ 批量保存 {len(table_records)} 个向量到Hologres")
                return True

            except Exception as e:
                trans.rollback()
                print(f"   ❌ 保存向量失败: {e}")
                return False

    def save_embeddings_incrementally(self, model, table_records: List[Dict],
                                      vector_texts: List[str]) -> Dict[str, int]:
        """增量更新向量：只对新增/修改的表做 encode 和 UPSERT，对已删除的表清空对应列

        Args:
            model: 向量模型实例
            table_records: 当前所有表的元数据
            vector_texts: 与 table_records 对应的 vector_text

        Returns:
            统计字典：{'new': N, 'changed': N, 'unchanged': N, 'removed': N}
        """
        embedding_col = self._embedding_col(
            'api' if (hasattr(model, 'name') and model.name.startswith('api:')) else 'local'
        )
        stats = {'new': 0, 'changed': 0, 'unchanged': 0, 'removed': 0}

        # 1. 加载现有 (schema, table) -> (text_hash, id, has_col) 快照
        existing = {}
        try:
            with self.engine.connect() as conn:
                existing_rows = conn.execute(text(f"""
                    SELECT COALESCE(schema_name,''), table_name, text_hash, id, {embedding_col} IS NOT NULL AS has_col
                    FROM knowledge.table_embeddings
                    WHERE db_name = :db_name
                """), {"db_name": self.db_name}).fetchall()
                for r in existing_rows:
                    existing[(r[0], r[1])] = (r[2], r[3], r[4])
        except Exception as e:
            print(f"   ⚠️ 加载现有向量快照失败，回退到全量重建: {e}")
            embeddings_list = []
            batch_size = 50
            for i in range(0, len(vector_texts), batch_size):
                batch = vector_texts[i:i+batch_size]
                emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
                embeddings_list.extend(emb)
            embeddings = np.array(embeddings_list)
            self.save_embeddings_to_holo(table_records, embeddings, embedding_col=embedding_col)
            stats['new'] = len(table_records)
            return stats

        # 2. 分类：新增 / 修改 / 不变 / 移除
        new_records, new_texts = [], []
        changed_records, changed_texts = [], []
        current_keys = set()

        for record, vec_text in zip(table_records, vector_texts):
            text_hash = hashlib.md5(vec_text.encode()).hexdigest()
            key = (record.get('schema') or '', record['table_name'])
            current_keys.add(key)
            record['_vector_text'] = vec_text
            record['_text_hash'] = text_hash
            old_hash, row_id, has_col = existing.get(key, (None, None, False))

            if row_id is None:
                new_records.append(record)
                new_texts.append(vec_text)
            elif old_hash != text_hash or not has_col:
                record['_row_id'] = row_id
                changed_records.append(record)
                changed_texts.append(vec_text)

        removed_keys = set(existing.keys()) - current_keys

        stats['new'] = len(new_records)
        stats['changed'] = len(changed_records)
        stats['unchanged'] = len(table_records) - stats['new'] - stats['changed']
        stats['removed'] = len(removed_keys)

        print(f"   ├─ 增量分析: 新增 {stats['new']}, 修改 {stats['changed']}, "
              f"不变 {stats['unchanged']}, 移除 {stats['removed']}")

        if stats['new'] == 0 and stats['changed'] == 0 and stats['removed'] == 0:
            print(f"   ✅ 向量库已是最新，无需更新")
            return stats

        # 3. 只对新增 + 修改的表 encode
        to_encode_records = new_records + changed_records
        to_encode_texts = new_texts + changed_texts

        embeddings_list = []
        if to_encode_texts:
            print(f"   ├─ 生成 {len(to_encode_texts)} 个向量...")
            batch_size = 50
            for i in range(0, len(to_encode_texts), batch_size):
                batch = to_encode_texts[i:i+batch_size]
                batch_emb = model.encode(batch, convert_to_numpy=True,
                                         show_progress_bar=False, normalize_embeddings=True)
                embeddings_list.extend(batch_emb)

        # 4. 写库
        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                # 4a. 移除的表：把该模型的 embedding 列置 NULL
                for schema_name, table_name in removed_keys:
                    conn.execute(text(f"""
                        UPDATE knowledge.table_embeddings
                        SET {embedding_col} = NULL, updated_at = NOW()
                        WHERE db_name = :db_name
                          AND COALESCE(schema_name, '') = :schema_name
                          AND table_name = :table_name
                    """), {
                        "db_name": self.db_name,
                        "schema_name": schema_name,
                        "table_name": table_name,
                    })

                # 4b. 批量 UPDATE 已有行
                changed_with_id = [r for r in changed_records if '_row_id' in r]
                if changed_with_id:
                    update_params = []
                    for record, embedding in zip(changed_with_id, embeddings_list[len(new_records):]):
                        update_params.append({
                            "id": record['_row_id'],
                            "embedding": embedding.tolist(),
                            "text_hash": record['_text_hash'],
                            "schema_hash": record['_text_hash'],
                            "vector_text": record['_vector_text'],
                            "table_comment": record.get('table_comment', ''),
                            "column_info": json.dumps(record.get('_columns_json', [])),
                        })

                    conn.execute(text(f"""
                        UPDATE knowledge.table_embeddings
                        SET {embedding_col} = :embedding,
                            text_hash = :text_hash,
                            schema_hash = :schema_hash,
                            vector_text = :vector_text,
                            table_comment = :table_comment,

                            column_info = CAST(:column_info AS jsonb),
                            updated_at = NOW()
                        WHERE id = :id
                    """), update_params)

                # 4c. 批量 INSERT 新行
                if new_records:
                    insert_params = []
                    for record, embedding in zip(new_records, embeddings_list[:len(new_records)]):
                        insert_params.append({
                            "db_name": self.db_name,
                            "schema_name": record.get('schema'),
                            "table_name": record['table_name'],
                            "table_comment": record.get('table_comment', ''),
                            "column_info": json.dumps(record.get('_columns_json', [])),
                            "vector_text": record['_vector_text'],
                            "embedding": embedding.tolist(),
                            "text_hash": record['_text_hash'],
                            "schema_hash": record['_text_hash'],
                        })
                    conn.execute(text(f"""
                        INSERT INTO knowledge.table_embeddings
                            (db_name, schema_name, table_name, table_comment, column_info,
                             vector_text, {embedding_col}, text_hash, schema_hash, updated_at)
                        VALUES
                            (:db_name, :schema_name, :table_name, :table_comment, CAST(:column_info AS jsonb),
                             :vector_text, :embedding, :text_hash, :schema_hash, NOW())
                    """), insert_params)

                trans.commit()
                _vectors_cache.invalidate(f"vectors:{self.db_name}")
                print(f"   ✅ 增量更新完成")

                self._update_schema_fingerprint(conn, table_records)

                return stats

            except Exception as e:
                trans.rollback()
                print(f"   ❌ 增量更新失败: {e}")
                raise

    def _update_schema_fingerprint(self, conn, table_records: List[Dict]):
        """更新 schema fingerprint 到元数据表"""
        try:
            # 计算全库表结构的 fingerprint
            fingerprint = self._compute_schema_fingerprint(table_records)

            # 元数据表需提前通过 sql/create_db_metadata.sql 创建

            # 插入或更新 fingerprint（DML 单独事务）
            conn.execute(text("""
                INSERT INTO knowledge.db_metadata (db_name, schema_fingerprint, last_updated)
                VALUES (:db_name, :fingerprint, NOW())
                ON CONFLICT (db_name) DO UPDATE SET
                    schema_fingerprint = EXCLUDED.schema_fingerprint,
                    last_updated = NOW()
            """), {"db_name": self.db_name, "fingerprint": fingerprint})
            conn.commit()
        except Exception as e:
            print(f"   ⚠️ 更新 schema fingerprint 失败: {e}")

    def _compute_schema_fingerprint(self, table_records: List[Dict]) -> str:
        """计算全库表结构的 fingerprint（所有表的 vector_text 拼接后 hash）"""
        # 按 schema.table 排序确保稳定性
        sorted_records = sorted(table_records, key=lambda r: f"{r.get('schema','')}.{r['table_name']}")
        combined = "\n".join(r.get('_vector_text', '') for r in sorted_records)
        return hashlib.md5(combined.encode()).hexdigest()

    def _fill_missing_knowledge_vectors(self):
        """补填知识库/名词中两个模型列 embedding 为空的记录"""
        self._ensure_embedding_columns()
        from .embedding_client import get_embedding_model

        for provider in ('local', 'api'):
            col = self._embedding_col(provider)
            model = get_embedding_model(provider)

            for table, id_field, text_fields, save_fn in [
                ('knowledge.db_knowledge', 'id', ('question', 'sql'),
                 lambda m, rid, *vals: self.save_single_knowledge_vector(m, rid, *vals)),
                ('knowledge.business_glossary', 'id', ('term', 'definition'),
                 lambda m, rid, *vals: self.save_single_glossary_vector(m, rid, *vals)),
            ]:
                with self.engine.connect() as conn:
                    missing = conn.execute(text(f"""
                        SELECT id, {', '.join(text_fields)} FROM {table}
                        WHERE db_name = :d AND {col} IS NULL
                    """), {"d": self.db_name}).fetchall()
                if missing:
                    print(f"   ├─ 补填 {len(missing)} 条 {table} 向量 ({provider})...")
                    for row in missing:
                        save_fn(model, row[0], row[1], row[2])

    def _cleanup_excluded_schemas(self):
        """删除已排除 schema 的行（如 hg_recyclebin）"""
        excluded = ['hg_recyclebin']
        for schema in excluded:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    DELETE FROM knowledge.table_embeddings
                    WHERE db_name = :d AND schema_name = :schema
                """), {"d": self.db_name, "schema": schema})
                if result.rowcount > 0:
                    print(f"   🗑️  清理 {self.db_name} 中 schema={schema} 的行: {result.rowcount} 条")
                    conn.commit()
                    _vectors_cache.invalidate(f"vectors:{self.db_name}")

    def _check_provider_vectors(self, table_records, vector_texts, provider: str):
        """检查并增量更新指定模型的向量列"""
        col = self._embedding_col(provider)
        with self.engine.connect() as conn:
            null_count = conn.execute(text(f"""
                SELECT COUNT(*) FROM knowledge.table_embeddings
                WHERE db_name = :d AND {col} IS NULL
            """), {"d": self.db_name}).fetchone()[0]
        if null_count > 0:
            print(f"   🔄 [{self.db_name}] {col}({provider}) 有空值，增量更新")
            self._rebuild_vectors(table_records, vector_texts, provider)

    def check_and_update_vectors(self):
        """检查表结构是否变化，变化则自动重建向量；同时补填知识库/名词的空向量"""
        try:
            # 获取当前表结构
            table_records, vector_texts = self.get_vector_texts()
            if not table_records:
                print(f"   ⚠️ [{self.db_name}] 无表结构，跳过检查")
                return

            current_fingerprint = self._compute_schema_fingerprint(table_records)

            # 查询上次保存的 fingerprint
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT schema_fingerprint FROM knowledge.db_metadata
                    WHERE db_name = :db_name
                """), {"db_name": self.db_name})
                row = result.fetchone()

            schema_changed = row is None or row[0] != current_fingerprint
            if schema_changed:
                print(f"   🔄 [{self.db_name}] {'首次建立' if row is None else '表结构变化，'}两个模型都重建")
                # 对两个模型都跑全量重建
                self._rebuild_vectors(table_records, vector_texts, 'local')
                self._rebuild_vectors(table_records, vector_texts, 'api')
            else:
                print(f"   ✅ [{self.db_name}] 表结构无变化，按列检查空值")
                self._check_provider_vectors(table_records, vector_texts, 'local')
                self._check_provider_vectors(table_records, vector_texts, 'api')

            # 补填知识库/名词的空向量（新增的数据）
            self._fill_missing_knowledge_vectors()

            # 清理已排除 schema 的行
            self._cleanup_excluded_schemas()

        except Exception as e:
            print(f"   ❌ [{self.db_name}] 检查向量失败: {e}")

    def _rebuild_vectors(self, table_records: List[Dict], vector_texts: List[str],
                         provider: str = None):
        """更新表结构向量（走增量路径），只重建指定模型的列"""
        from .embedding_client import get_embedding_model

        print(f"   ├─ 加载向量模型 (provider={provider or 'default'})...")
        model = get_embedding_model(provider)
        self.save_embeddings_incrementally(model, table_records, vector_texts)
        print(f"   ✅ [{self.db_name}] 表结构向量同步完成")

    # 类级别记录：已经验证过 SQL 路径不可用的 db 名（避免每次查询都先试 SQL 再 fallback）
    _sql_path_disabled = set()
    _sql_path_lock = threading.Lock()

    def vector_search_in_holo(self, query_embedding: List[float], top_k: int = 10,
                              schema_filter: Optional[str] = None,
                              embedding_col: str = 'local_embedding') -> List[Dict]:
        """向量检索：优先走 Hologres Proxima 索引（快），失败时 fallback 到 numpy。

        Args:
            embedding_col: 要搜索的向量列名（local_embedding / doubao_embedding）
        """
        if self.db_name not in self._sql_path_disabled:
            try:
                results = self._vector_search_via_holo_sql(query_embedding, top_k, schema_filter, embedding_col)
                return results
            except Exception as e:
                with self._sql_path_lock:
                    self._sql_path_disabled.add(self.db_name)
                print(f"   ⚠️ Hologres Proxima 路径不可用（已切换到 numpy fallback）: {e}")

        return self._vector_search_via_numpy(query_embedding, top_k, schema_filter, embedding_col)

    def _vector_search_via_holo_sql(self, query_embedding: List[float], top_k: int,
                                    schema_filter: Optional[str] = None,
                                    embedding_col: str = 'local_embedding') -> List[Dict]:
        """使用 Hologres Proxima 索引查询"""
        embedding_str = '{' + ','.join(str(x) for x in query_embedding) + '}'
        params = {"db_name": self.db_name, "query_embedding": embedding_str, "top_k": top_k}
        clauses = [f"{embedding_col} IS NOT NULL"]

        if schema_filter:
            schemas = [s.strip() for s in schema_filter.split(',') if s.strip()]
            if schemas:
                clauses.append("schema_name = ANY(:schemas)")
                params["schemas"] = schemas

        where_clause = " AND ".join(clauses)

        sql = f"""
        SELECT
            schema_name,
            table_name,
            table_comment,
            column_info,
            vector_text,
            pm_approx_squared_euclidean_distance({embedding_col}, CAST(:query_embedding AS float4[])) AS similarity_score
        FROM knowledge.table_embeddings
        WHERE db_name = :db_name AND {where_clause}
        ORDER BY similarity_score ASC
        LIMIT :top_k
        """

        with self.engine.connect() as conn:
            try:
                conn.execute(text("SET hg_computing_resource = 'serverless'"))
            except Exception:
                pass

            result = conn.execute(text(sql), params)
            rows = result.fetchall()

        if not rows:
            return []

        results = []
        for i, row in enumerate(rows, 1):
            results.append({
                'schema': row[0],
                'table_name': row[1],
                'table_comment': row[2] or '',
                '_columns_json': row[3] if row[3] else [],
                '_vector_text': row[4] or '',
                '_similarity_score': float(row[5]),
                '_rank': i,
            })
        return results

    def _vector_search_via_numpy(self, query_embedding: List[float], top_k: int,
                                 schema_filter: Optional[str] = None,
                                 embedding_col: str = 'local_embedding') -> List[Dict]:
        """fallback：从 Hologres 拉出全部 embedding，在 Python 内存用 numpy 算 squared euclidean distance"""
        records, embeddings_matrix = self._load_all_vectors_cached(embedding_col)
        if not records:
            return []

        try:
            if schema_filter:
                schemas = set(s.strip().lower() for s in schema_filter.split(',') if s.strip())
                if schemas:
                    keep = [i for i, r in enumerate(records) if (r.get('schema') or '').lower() in schemas]
                    if not keep:
                        print(f"   ⚠️ schema 过滤后无可用向量（filter={schema_filter}）")
                        return []
                    embeddings_matrix = embeddings_matrix[keep]
                    records = [records[i] for i in keep]

            query_vec = np.array(query_embedding, dtype=np.float32)
            diff = embeddings_matrix - query_vec
            distances = np.einsum('ij,ij->i', diff, diff)

            n = len(distances)
            k = min(top_k, n)
            if k == n:
                top_indices = np.argsort(distances)
            else:
                cand = np.argpartition(distances, k)[:k]
                top_indices = cand[np.argsort(distances[cand])]

            results = []
            for rank, idx in enumerate(top_indices, 1):
                rec = records[idx]
                results.append({
                    'schema': rec['schema'],
                    'table_name': rec['table_name'],
                    'table_comment': rec['table_comment'],
                    '_columns_json': rec['column_info'],
                    '_vector_text': rec['vector_text'],
                    '_similarity_score': float(distances[idx]),
                    '_rank': rank,
                })
            return results
        except Exception as e:
            print(f"   ❌ numpy 向量检索失败: {e}")
            return []

    def _search_inline_table(self, table: str, query: str, top_k: int,
                              id_field: str, text_fields: List[str],
                              provider: str = None) -> List[Dict]:
        """通用：对双列 embedding 的表做向量检索（numpy fallback）"""
        from .embedding_client import get_embedding_model
        model = get_embedding_model(provider)
        col = self._embedding_col(
            'api' if (hasattr(model, 'name') and model.name.startswith('api:')) else 'local'
        )

        with self.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT id, {', '.join(text_fields)}, {col}
                FROM {table}
                WHERE db_name = :d AND {col} IS NOT NULL
            """), {"d": self.db_name}).fetchall()

        if not rows:
            return []

        records = []
        embeddings = []
        for r in rows:
            records.append({f: getattr(r, f, '') for f in ['id'] + text_fields})
            embeddings.append(r[-1])

        query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        emb_matrix = np.array(embeddings, dtype=np.float32)
        diff = emb_matrix - query_emb
        distances = np.einsum('ij,ij->i', diff, diff)

        n = len(distances)
        k = min(top_k, n)
        if k == n:
            top_indices = np.argsort(distances)
        else:
            cand = np.argpartition(distances, k)[:k]
            top_indices = cand[np.argsort(distances[cand])]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            rec = records[idx]
            results.append({
                '_id': rec['id'],
                '_score': float(distances[idx]),
                '_rank': rank,
                **{f: rec[f] for f in text_fields},
            })
        return results

    def knowledge_vector_search(self, query: str, top_k: int = 5,
                                 provider: str = None) -> List[Dict]:
        """向量检索知识库（SQL 知识）"""
        return self._search_inline_table(
            'knowledge.db_knowledge', query, top_k,
            id_field='id', text_fields=['question', 'sql'], provider=provider,
        )

    def glossary_vector_search(self, query: str, top_k: int = 5,
                                provider: str = None) -> List[Dict]:
        """向量检索业务名词"""
        return self._search_inline_table(
            'knowledge.business_glossary', query, top_k,
            id_field='id', text_fields=['term', 'definition'], provider=provider,
        )

    def _load_all_vectors_cached(self, embedding_col: str = 'local_embedding'):
        """从 Hologres 拉取指定列的向量到内存，带 TTL 缓存"""
        cache_key = f"vectors:{self.db_name}:{embedding_col}"
        cached = _vectors_cache.get(cache_key)
        if cached is not None:
            return cached

        sql = f"""
        SELECT schema_name, table_name, table_comment, column_info, vector_text, {embedding_col}
        FROM knowledge.table_embeddings
        WHERE db_name = :db_name AND {embedding_col} IS NOT NULL
        """

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), {"db_name": self.db_name})
                rows = result.fetchall()
        except Exception as e:
            print(f"   ❌ 加载向量失败: {e}")
            return [], np.array([])

        if not rows:
            return [], np.array([])

        records = []
        embedding_lists = []
        for row in rows:
            records.append({
                'schema': row[0],
                'table_name': row[1],
                'table_comment': row[2] or '',
                'column_info': row[3] if row[3] else [],
                'vector_text': row[4] or '',
            })
            embedding_lists.append(row[5])

        embeddings_matrix = np.array(embedding_lists, dtype=np.float32)
        result = (records, embeddings_matrix)
        _vectors_cache.set(cache_key, result)
        print(f"   ├─ 加载 {len(records)} 个表向量到内存（{embedding_col}, {embeddings_matrix.nbytes / 1024:.1f} KB）")
        return result

    def check_holo_vectors_exist(self) -> bool:
        sql = """
        SELECT COUNT(*) FROM knowledge.table_embeddings WHERE db_name = :db_name
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {"db_name": self.db_name})
            count = result.fetchone()[0]
            return count > 0

    def get_holo_vectors_count(self) -> int:
        sql = """
        SELECT COUNT(*) FROM knowledge.table_embeddings WHERE db_name = :db_name
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {"db_name": self.db_name})
            return result.fetchone()[0]

    def get_holo_vectors_count_by_model(self, provider: str) -> int:
        """查询指定模型在 table_embeddings 中的向量数"""
        col = 'doubao_embedding' if provider == 'api' else 'local_embedding'
        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT COUNT(*) FROM knowledge.table_embeddings
                WHERE db_name = :db_name AND {col} IS NOT NULL
            """), {"db_name": self.db_name})
            return result.fetchone()[0]

    # ========== 知识库/名词向量操作方法 ==========

    def _ensure_embedding_columns(self):
        """确保 db_knowledge 和 business_glossary 表有双列 embedding 字段"""
        for table in ['knowledge.db_knowledge', 'knowledge.business_glossary']:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text(f"""
                        ALTER TABLE {table}
                        ADD COLUMN IF NOT EXISTS local_embedding REAL[]
                    """))
                    conn.execute(text(f"""
                        ALTER TABLE {table}
                        ADD COLUMN IF NOT EXISTS doubao_embedding REAL[]
                    """))
                    conn.commit()
            except Exception as e:
                print(f"   ⚠️ 为 {table} 添加 embedding 字段失败: {e}")

    def save_single_knowledge_vector(self, model, knowledge_id: int, question: str, sql: str):
        """对单条知识条目生成向量并直接更新到 db_knowledge 表"""
        self._ensure_embedding_columns()
        col = self._embedding_col(
            'api' if (hasattr(model, 'name') and model.name.startswith('api:')) else 'local'
        )
        vector_text = f"问题: {question}\nSQL: {sql}"
        embedding = model.encode([vector_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        embedding_list = embedding.tolist()

        with self.engine.connect() as conn:
            try:
                conn.execute(text(f"""
                    UPDATE knowledge.db_knowledge
                    SET {col} = :embedding, updated_at = NOW()
                    WHERE id = :id AND db_name = :db_name
                """), {
                    "id": knowledge_id,
                    "db_name": self.db_name,
                    "embedding": embedding_list,
                })
                conn.commit()
            except Exception as e:
                print(f"   ❌ 保存知识条目向量失败: {e}")

    def save_single_glossary_vector(self, model, glossary_id: int, term: str, definition: str):
        """对单条业务名词生成向量并直接更新到 business_glossary 表"""
        self._ensure_embedding_columns()
        col = self._embedding_col(
            'api' if (hasattr(model, 'name') and model.name.startswith('api:')) else 'local'
        )
        vector_text = f"{term}: {definition}"
        embedding = model.encode([vector_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        embedding_list = embedding.tolist()

        with self.engine.connect() as conn:
            try:
                conn.execute(text(f"""
                    UPDATE knowledge.business_glossary
                    SET {col} = :embedding, updated_at = NOW()
                    WHERE id = :id AND db_name = :db_name
                """), {
                    "id": glossary_id,
                    "db_name": self.db_name,
                    "embedding": embedding_list,
                })
                conn.commit()
            except Exception as e:
                print(f"   ❌ 保存业务名词向量失败: {e}")

    def rebuild_knowledge_vectors(self):
        """重建所有知识库和业务名词的向量（两个模型都跑）"""
        self._ensure_embedding_columns()
        from .repos import SQLKnowledgeRepo, GlossaryRepo

        for provider in ('local', 'api'):
            from .embedding_client import get_embedding_model
            model = get_embedding_model(provider)
            print(f"   ├─ 重建知识库/名词向量 (模型: {model.name})")

            knowledge_repo = SQLKnowledgeRepo(self.db_name)
            for item in knowledge_repo.list():
                self.save_single_knowledge_vector(model, item['id'], item['question'], item['sql'])
            print(f"   ✅ 已更新知识库向量 ({provider})")

            glossary_repo = GlossaryRepo(self.db_name)
            for item in glossary_repo.list():
                self.save_single_glossary_vector(model, item['id'], item['term'], item['definition'])
            print(f"   ✅ 已更新业务名词向量 ({provider})")

    # ========== 表结构获取方法 ==========

    @monitor_function
    def get_table_schema(self) -> Dict:
        cache_key = f"schema:{self.db_name}"
        cached = _schema_cache.get(cache_key)
        if cached is not None:
            print(f"    ├─ 命中表结构缓存: {self.db_name}")
            return cached

        print(f"    ├─ 加载表结构: {self.db_name}")

        db_config = get_database_config(self.db_name)
        if not db_config:
            return {'tables': {}}

        if db_config['type'] == 'postgresql':
            result = self._get_postgresql_schema()
        elif db_config['type'] == 'sqlite':
            result = self._get_sqlite_schema()
        else:
            result = {'tables': {}}

        _schema_cache.set(cache_key, result)
        return result

    def _get_postgresql_schema(self) -> Dict:
        print(f"    ├─ 获取PostgreSQL/Hologres表结构...")

        excluded_schemas = "'information_schema', 'pg_catalog', 'knowledge', 'hg_recyclebin'"

        query = f"""
        WITH
        table_comments AS (
            SELECT
                n.nspname as schema_name,
                c.relname as table_name,
                COALESCE(pg_catalog.obj_description(c.oid, 'pg_class'), '') as table_comment
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind IN ('r', 'p', 'v', 'm')
                AND n.nspname NOT IN ({excluded_schemas})
                and c.relname NOT LIKE '%%_middle%%'
                AND n.nspname NOT LIKE 'pg_%%'
                AND has_schema_privilege(n.nspname, 'USAGE')
        ),
        column_comments AS (
            SELECT
                n.nspname as schema_name,
                c.relname as table_name,
                a.attname as column_name,
                a.attnum as column_order,
                COALESCE(pg_catalog.col_description(c.oid, a.attnum), '') as column_comment,
                pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE a.attnum > 0
                AND NOT a.attisdropped
                AND c.relkind IN ('r', 'p', 'v', 'm')
                AND n.nspname NOT IN ({excluded_schemas})
                and c.relname NOT LIKE '%%_middle%%'
                AND n.nspname NOT LIKE 'pg_%%'
                AND has_schema_privilege(n.nspname, 'USAGE')
        ),
        table_columns AS (
            SELECT
                schema_name,
                table_name,
                string_agg(
                    CASE
                        WHEN column_comment != '' THEN column_name || '[' || data_type || '](' || column_comment || ')'
                        ELSE column_name || '[' || data_type || ']'
                    END,
                    ', '
                    ORDER BY column_order
                ) as columns_text,
                string_agg(
                    CASE
                        WHEN column_comment != '' THEN '    ' || column_name || ' [' || data_type || '] (' || column_comment || ')'
                        ELSE '    ' || column_name || ' [' || data_type || ']'
                    END,
                    E'\n'
                    ORDER BY column_order
                ) as columns_formatted,
                string_agg(
                    CASE
                        WHEN column_comment != '' THEN column_comment
                        WHEN column_name ~ '[一-龥]' THEN column_name
                        ELSE NULL
                    END,
                    '、'
                    ORDER BY column_order
                ) as columns_for_vector,
                COUNT(*) as column_count,
                jsonb_agg(
                    jsonb_build_object('name', column_name, 'comment', column_comment, 'type', data_type)
                    ORDER BY column_order
                ) as columns_json
            FROM column_comments
            GROUP BY schema_name, table_name
        )
        SELECT
            tc.schema_name,
            tc.table_name,
            tc.table_comment,
            tcl.columns_text,
            tcl.columns_formatted,
            tcl.column_count,
            tcl.columns_json,
            TRIM(CONCAT(
                tc.schema_name, '.', tc.table_name,
                CASE WHEN tc.table_comment != '' THEN ' ' || tc.table_comment ELSE '' END,
                CASE WHEN tcl.columns_for_vector != '' THEN ' 字段: ' || tcl.columns_for_vector ELSE '' END
            )) as vector_text
        FROM table_comments tc
        JOIN table_columns tcl ON tc.schema_name = tcl.schema_name AND tc.table_name = tcl.table_name
        ORDER BY tc.schema_name, tc.table_name;
        """

        with self.engine.connect() as conn:
            try:
                conn.execute(text("ROLLBACK"))
            except Exception:
                pass

            result = conn.execute(text(query))
            rows = result.fetchall()

        print(f"    ├─ 获取到 {len(rows)} 个表")

        schemas = {}
        table_records = []
        all_vector_texts = []

        for row in rows:
            schema_name = row[0]
            table_name = row[1]
            table_comment = row[2]
            columns_text = row[3]
            columns_formatted = row[4]
            column_count = row[5]
            columns_json = row[6]
            vector_text = row[7]

            if schema_name not in schemas:
                schemas[schema_name] = {'tables': {}}

            schemas[schema_name]['tables'][table_name] = {
                '_table_comment': table_comment,
                '_columns_formatted': columns_formatted,
                '_column_count': column_count,
                '_vector_text': vector_text,
                '_columns_json': columns_json
            }

            col_names = [col.split('(')[0].strip() for col in columns_text.split(', ')] if columns_text else []
            table_records.append({
                'schema': schema_name,
                'table_name': table_name,
                'table_comment': table_comment,
                'columns': col_names,
                'column_count': column_count,
                '_vector_text': vector_text,
                '_columns_json': columns_json
            })
            all_vector_texts.append(vector_text)

        return {'schemas': schemas, 'table_records': table_records, 'vector_texts': all_vector_texts}

    def _get_sqlite_schema(self) -> Dict:
        print(f"    ├─ 获取SQLite表结构...")

        with self.engine.connect() as conn:
            tables_result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
            tables = {}
            for table_name in [row[0] for row in tables_result.fetchall()]:
                pragma_result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = {row[1]: row[2] for row in pragma_result.fetchall()}
                tables[table_name] = columns

        return {'tables': tables}

    def get_table_list(self) -> List[Dict]:
        schema_data = self.get_table_schema()
        table_list = []

        if 'schemas' in schema_data:
            for schema, schema_info in schema_data['schemas'].items():
                for table_name, table_info in schema_info.get('tables', {}).items():
                    table_list.append({
                        'schema': schema,
                        'table_name': table_name,
                        'table_comment': table_info.get('_table_comment', ''),
                        'column_count': table_info.get('_column_count', 0),
                        'columns': [col['name'] for col in table_info.get('_columns_json', [])]
                    })
        else:
            for table_name, table_info in schema_data.get('tables', {}).items():
                table_list.append({
                    'schema': None,
                    'table_name': table_name,
                    'table_comment': '',
                    'column_count': len(table_info),
                    'columns': list(table_info.keys())
                })

        return table_list

    def get_vector_texts(self) -> Tuple[List[Dict], List[str]]:
        schema_data = self.get_table_schema()
        if 'table_records' in schema_data and 'vector_texts' in schema_data:
            return schema_data['table_records'], schema_data['vector_texts']
        return [], []

    def get_table_schema_by_name(self, table_name: str) -> str:
        if not table_name:
            return ""

        schema_data = self.get_table_schema()

        if '.' in table_name:
            schema, table = table_name.split('.', 1)
        else:
            schema = None
            table = table_name

        if 'schemas' in schema_data:
            if schema and schema in schema_data['schemas']:
                table_info = schema_data['schemas'][schema]['tables'].get(table)
                if table_info:
                    columns_formatted = table_info.get('_columns_formatted')
                    table_comment = table_info.get('_table_comment', '')
                    header = f"表: {schema}.{table}"
                    if table_comment:
                        header += f" (注释: {table_comment})"
                    if columns_formatted:
                        return header + "\n列:\n" + columns_formatted
                    else:
                        return header

            for schema_name, schema_info in schema_data['schemas'].items():
                table_info = schema_info['tables'].get(table)
                if table_info:
                    columns_formatted = table_info.get('_columns_formatted')
                    table_comment = table_info.get('_table_comment', '')
                    header = f"表: {schema_name}.{table}"
                    if table_comment:
                        header += f" (注释: {table_comment})"
                    if columns_formatted:
                        return header + "\n列:\n" + columns_formatted
                    else:
                        return header

        return f"未找到表: {table_name}"

    def get_formatted_schema(self, selected_tables: Optional[List[str]] = None, schema_filter: Optional[str] = None) -> str:
        schema_data = self.get_table_schema()

        if selected_tables:
            return self._format_selected_tables(schema_data, selected_tables)
        else:
            return self._format_all_tables(schema_data, schema_filter=schema_filter)

    def _format_all_tables(self, schema_data: Dict, schema_filter: Optional[str] = None) -> str:
        formatted = []
        filter_set = None
        if schema_filter:
            filter_set = set(s.strip() for s in schema_filter.split(',') if s.strip())

        if 'schemas' in schema_data:
            for schema_name, schema_info in schema_data['schemas'].items():
                if filter_set and schema_name not in filter_set:
                    continue
                for table_name, table_info in schema_info.get('tables', {}).items():
                    columns_formatted = table_info.get('_columns_formatted')
                    table_comment = table_info.get('_table_comment', '')

                    header = f"表名: {schema_name}.{table_name}"
                    if table_comment:
                        header += f" 表注释: {table_comment}"

                    if columns_formatted:
                        formatted.append(header + "\n列:\n" + columns_formatted)
                    else:
                        formatted.append(header)

        return "\n\n".join(formatted) if formatted else "无可用表结构"

    def _format_selected_tables(self, schema_data: Dict, selected_tables: List[str]) -> str:
        formatted = []
        found_count = 0

        for full_name in selected_tables:
            if '.' in full_name:
                schema_name, table_name = full_name.split('.', 1)
            else:
                schema_name = None
                table_name = full_name

            table_info = None

            if 'schemas' in schema_data:
                if schema_name and schema_name in schema_data['schemas']:
                    table_info = schema_data['schemas'][schema_name]['tables'].get(table_name)
                else:
                    for s_name, s_info in schema_data['schemas'].items():
                        if table_name in s_info['tables']:
                            table_info = s_info['tables'][table_name]
                            schema_name = s_name
                            break

            if table_info:
                columns_formatted = table_info.get('_columns_formatted')
                table_comment = table_info.get('_table_comment', '')
                header = f"表名: {schema_name}.{table_name}"
                if table_comment:
                    header += f" 表注释: {table_comment}"
                if columns_formatted:
                    formatted.append(header + "\n列:\n" + columns_formatted)
                else:
                    formatted.append(header)
                found_count += 1
            else:
                print(f"   ⚠️ 未找到表: {full_name}")

        if not formatted:
            return f"未找到指定的表: {', '.join(selected_tables)}"

        print(f"   ✅ 成功格式化 {found_count}/{len(selected_tables)} 个表")
        return "\n\n".join(formatted)


# ========== 后台监控线程 ==========

_monitor_thread = None
_monitor_lock = threading.Lock()

def start_vector_monitor(db_names: List[str]):
    """启动后台监控线程，每天凌晨 1:00 检查表结构变化并自动更新向量

    Args:
        db_names: 要监控的数据库名称列表
    """
    from datetime import datetime, timedelta
    global _monitor_thread

    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            print("   ⚠️ 向量监控线程已在运行")
            return

        def monitor_loop():
            print(f"🔍 向量监控线程启动，每天凌晨 1:00 执行一次")

            while True:
                now = datetime.now()
                next_run = now.replace(hour=1, minute=0, second=0, microsecond=0)
                if now >= next_run:
                    next_run = next_run + timedelta(days=1)
                sleep_seconds = (next_run - now).total_seconds()
                print(f"   ├─ 距下次检查还有 {sleep_seconds / 3600:.1f} 小时 ({next_run.strftime('%Y-%m-%d %H:%M')})")
                time.sleep(sleep_seconds)

                print(f"\n🔍 [{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始每日向量检查...")
                for db_name in db_names:
                    try:
                        kb = KnowledgeBase(db_name)
                        kb.check_and_update_vectors()
                    except Exception as e:
                        print(f"   ❌ [{db_name}] 检查失败: {e}")

        _monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name="VectorMonitor")
        _monitor_thread.start()
        print(f"   ✅ 向量监控线程已启动（监控 {len(db_names)} 个数据库，每天 1:00 自动更新）")

