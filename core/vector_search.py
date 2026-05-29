# core/vector_search.py - 表结构向量检索（带模型缓存）
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .knowledge import KnowledgeBase
from .utils import SENTENCE_TRANSFORMER_MODEL, monitor_function


class TableSchemaSearcher:
    _model = None
    _model_lock = threading.Lock()

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            with cls._model_lock:
                if cls._model is None:
                    load_start = time.time()
                    cls._model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
                    load_duration = (time.time() - load_start) * 1000
                    print(f"    ├─ 加载向量模型: {load_duration:.2f} ms")
        return cls._model

    @classmethod
    @monitor_function
    def search(cls, db_name: str, query: str, top_k: int = 10,
               kb: KnowledgeBase = None, use_holo_index: bool = True,
               force_rebuild_vectors: bool = False,
               schema_filter: Optional[str] = None) -> List[Dict]:

        if not kb:
            kb = KnowledgeBase(db_name)

        if use_holo_index:
            vectors_count = kb.get_holo_vectors_count()

            if vectors_count == 0 or force_rebuild_vectors:
                table_records, vector_texts = kb.get_vector_texts()

                if not table_records:
                    return []

                model = cls._get_model()

                batch_size = 50
                all_embeddings = []
                for i in range(0, len(vector_texts), batch_size):
                    batch = vector_texts[i:i+batch_size]
                    batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
                    all_embeddings.extend(batch_embeddings)

                embeddings = np.array(all_embeddings)
                kb.save_embeddings_to_holo(table_records, embeddings)

        model = cls._get_model()
        query_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)[0]
        results = kb.vector_search_in_holo(query_emb.tolist(), top_k, schema_filter=schema_filter)

        if not results:
            return []

        print("\n" + "=" * 70)
        print("📊 表名按相似度排序结果（欧氏距离，越小越相似）")
        print("=" * 70)

        for i, r in enumerate(results, 1):
            table_display = f"{r['schema']}.{r['table_name']}" if r['schema'] else r['table_name']
            similarity_score = r.get('_similarity_score', 0)
            print(f"{i:2d}. 表名: {table_display:<50} 距离: {similarity_score:.4f}")

        print("=" * 70 + "\n")

        return results
