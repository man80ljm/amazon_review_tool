# core/representatives.py
from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def top_representatives(embeddings: np.ndarray, labels: np.ndarray, centers: np.ndarray, top_n: int = 5) -> Dict[int, List[int]]:
    # 返回每簇最接近中心的样本索引
    reps = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            reps[c] = []
            continue
        sims = cosine_similarity(embeddings[idx], centers[c].reshape(1, -1)).ravel()
        best = idx[np.argsort(sims)[::-1][:top_n]]
        reps[c] = best.tolist()
    return reps
