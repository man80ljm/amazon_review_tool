# core/robustness.py
from typing import Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def clustering_stability(embeddings: np.ndarray, k: int, runs: int = 5, random_state: int = 42) -> Dict[str, float]:
    # bootstrap: 对全量 N 做有放回抽样 -> 还原到 N 长度的标签（用抽样索引的标签填回去）
    n = embeddings.shape[0]
    label_runs = []

    rng = np.random.default_rng(random_state)
    for i in range(runs):
        idx = rng.integers(0, n, size=n)  # bootstrap indices
        emb_bs = embeddings[idx]
        km = KMeans(n_clusters=k, random_state=random_state + i, n_init="auto", max_iter=500)
        lab_bs = km.fit_predict(emb_bs)

        # 回填到原长度（同一位置对应 idx）
        lab_full = np.empty(n, dtype=int)
        lab_full[:] = -1
        lab_full[idx] = lab_bs
        # 对没被抽到的位置，简单用最近中心再分配（保证可比）
        if (lab_full == -1).any():
            centers = km.cluster_centers_
            missing = np.where(lab_full == -1)[0]
            # 余弦相似最大
            sims = emb_similarity(embeddings[missing], centers)
            lab_full[missing] = sims.argmax(axis=1)

        label_runs.append(lab_full)

    # 两两ARI
    aris = []
    for i in range(runs):
        for j in range(i+1, runs):
            aris.append(adjusted_rand_score(label_runs[i], label_runs[j]))

    return {
        "runs": float(runs),
        "ari_mean": float(np.mean(aris)) if aris else 0.0,
        "ari_min": float(np.min(aris)) if aris else 0.0,
        "ari_max": float(np.max(aris)) if aris else 0.0,
    }

def emb_similarity(A: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # 归一化后点积 = 余弦相似
    A2 = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    C2 = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    return A2 @ C2.T
