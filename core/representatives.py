# core/representatives.py
from typing import Dict, List, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def top_representatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: Union[np.ndarray, Dict[int, np.ndarray]],
    top_n: int = 5,
    noise_label: int | None = None
) -> Dict[int, List[int]]:
    # 返回每簇最接近中心的样本索引
    reps = {}
    for c in sorted(set(labels.tolist())):
        if noise_label is not None and int(c) == int(noise_label):
            continue
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            reps[c] = []
            continue
        center_vec = _get_center(centers, c)
        if center_vec is None:
            reps[c] = []
            continue
        sims = cosine_similarity(embeddings[idx], center_vec.reshape(1, -1)).ravel()
        best = idx[np.argsort(sims)[::-1][:top_n]]
        reps[c] = best.tolist()
    return reps


def _get_center(centers: Union[np.ndarray, Dict[int, np.ndarray]], label: int) -> np.ndarray | None:
    if isinstance(centers, dict):
        return centers.get(int(label))
    try:
        return centers[int(label)]
    except Exception:
        return None
