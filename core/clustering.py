# core/clustering.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

@dataclass
class KScanResult:
    k_to_inertia: Dict[int, float]
    k_to_silhouette: Dict[int, float]

def scan_k(embeddings: np.ndarray, k_min: int, k_max: int, random_state: int = 42) -> KScanResult:
    k_to_inertia = {}
    k_to_sil = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto", max_iter=500)
        labels = km.fit_predict(embeddings)
        k_to_inertia[k] = float(km.inertia_)
        if len(set(labels)) > 1:
            k_to_sil[k] = float(silhouette_score(embeddings, labels))
        else:
            k_to_sil[k] = -1.0
    return KScanResult(k_to_inertia, k_to_sil)

def fit_kmeans(embeddings: np.ndarray, k: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto", max_iter=500)
    labels = km.fit_predict(embeddings)
    centers = km.cluster_centers_
    return labels, centers
