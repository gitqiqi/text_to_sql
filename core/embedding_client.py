import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union

import numpy as np
import requests

from .utils import EMBEDDING_PROVIDER, SENTENCE_TRANSFORMER_MODEL

EMBEDDING_API_BASE = os.getenv('EMBEDDING_API_BASE', 'https://ark.cn-beijing.volces.com/api/v3')


class BaseEmbeddingModel:
    @property
    def name(self) -> str:
        return "unknown"

    def encode(self, texts: Union[str, List[str]], convert_to_numpy: bool = True,
               show_progress_bar: bool = False, normalize_embeddings: bool = True) -> np.ndarray:
        raise NotImplementedError


class LocalEmbeddingModel(BaseEmbeddingModel):
    @property
    def name(self) -> str:
        return self._model_id

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model_id = os.path.basename(os.getenv('SENTENCE_TRANSFORMER_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2'))
        self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True):
        return self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
        )


class ApiEmbeddingModel(BaseEmbeddingModel):
    @property
    def name(self) -> str:
        return f"api:{self.api_model_name}"

    def __init__(self):
        self.api_key = os.getenv('ARK_EMBEDDING_API_KEY') or os.getenv('ARK_API_KEY')
        if not self.api_key:
            raise ValueError("ARK_EMBEDDING_API_KEY or ARK_API_KEY is required when EMBEDDING_PROVIDER=api")
        self.api_model_name = os.getenv('ARK_EMBEDDING_MODEL', 'doubao-embedding-vision-251215')

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]

        def _call_api(text: str):
            resp = requests.post(
                f"{EMBEDDING_API_BASE}/embeddings/multimodal",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.api_model_name, "input": [{"type": "text", "text": text}]},
                timeout=120,
            )
            resp.raise_for_status()
            emb = resp.json()["data"]["embedding"]
            if normalize_embeddings:
                emb = self._normalize(emb)
            return emb

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_call_api, t) for t in texts]
            if show_progress_bar:
                from tqdm import tqdm
                futures = tqdm(futures, desc="API Embed")
            embeddings = [f.result() for f in futures]

        if convert_to_numpy:
            return np.array(embeddings, dtype=np.float32)
        return embeddings

    @staticmethod
    def _normalize(vec):
        norm = np.linalg.norm(vec)
        if norm > 0:
            return (np.array(vec) / norm).tolist()
        return vec


def get_embedding_model(provider: str = None) -> BaseEmbeddingModel:
    if provider is None:
        provider = EMBEDDING_PROVIDER
    if provider == 'api':
        model_name = os.getenv('ARK_EMBEDDING_MODEL', 'doubao-embedding-vision-251215')
        print(f"📦 向量模型: API ({model_name})")
        return ApiEmbeddingModel()
    print(f"📦 向量模型: 本地 ({SENTENCE_TRANSFORMER_MODEL})")
    return LocalEmbeddingModel()
