# core/embedding.py
from typing import Callable, Optional, List
import os
import numpy as np
from sentence_transformers import SentenceTransformer

ProgressCb = Optional[Callable[[int, int, str], None]]

class Embedder:
    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], cache_path: Optional[str] = None, progress: ProgressCb = None) -> np.ndarray:
        if cache_path and os.path.exists(cache_path):
            return np.load(cache_path)

        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, emb)

        if progress:
            progress(len(texts), len(texts), "Embedding done")
        return emb
