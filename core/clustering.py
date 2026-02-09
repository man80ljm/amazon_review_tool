# core/clustering.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

@dataclass
class KScanResult:
    k_to_inertia: Dict[int, float]
    k_to_silhouette: Dict[int, float]
    k_to_ch: Optional[Dict[int, float]] = None

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


def scan_k_agglomerative(
    embeddings: np.ndarray,
    k_min: int,
    k_max: int,
    linkage: str = "ward",
    metric: str = "euclidean",
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> KScanResult:
    k_to_sil = {}
    k_to_ch = {}

    # Ward only supports euclidean
    if str(linkage).lower() == "ward":
        metric = "euclidean"

    n = embeddings.shape[0]
    idx = None
    if sample_size is not None and sample_size > 0 and n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=int(sample_size), replace=False)
        X_eval = embeddings[idx]
    else:
        X_eval = embeddings

    for k in range(k_min, k_max + 1):
        model = _build_agglomerative(n_clusters=k, linkage=linkage, metric=metric)
        labels = model.fit_predict(embeddings)

        # Metrics computed on full set or sampled subset
        labels_use = labels if idx is None else labels[idx]

        if len(set(labels_use)) > 1:
            try:
                k_to_sil[k] = float(silhouette_score(X_eval, labels_use, metric=metric))
            except Exception:
                k_to_sil[k] = None
            try:
                k_to_ch[k] = float(calinski_harabasz_score(X_eval, labels_use))
            except Exception:
                k_to_ch[k] = None
        else:
            k_to_sil[k] = None
            k_to_ch[k] = None

    return KScanResult(k_to_inertia={}, k_to_silhouette=k_to_sil, k_to_ch=k_to_ch)


def run_clustering(
    embeddings: np.ndarray,
    method: str,
    params: Dict[str, Any],
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    method_key = (method or "KMeans").strip()
    method_norm = method_key.lower()
    start = time.time()

    if method_norm == "kmeans":
        k = int(params.get("n_clusters") or params.get("k") or 5)
        n_init = params.get("n_init", "auto")
        max_iter = int(params.get("max_iter", 500))
        tol = float(params.get("tol", 1e-4))
        init = params.get("init", "k-means++")

        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            init=init
        )
        labels = km.fit_predict(embeddings)
        centers = km.cluster_centers_

        meta = {
            "method": "KMeans",
            "method_params": {
                "n_clusters": k,
                "n_init": n_init,
                "max_iter": max_iter,
                "tol": tol,
                "init": init,
            },
            "centers": centers,
        }

    elif method_norm == "agglomerative":
        k = int(params.get("n_clusters") or params.get("k") or 5)
        linkage = str(params.get("linkage", "ward")).lower()
        metric = str(params.get("metric", "euclidean")).lower()
        if linkage == "ward":
            metric = "euclidean"

        model = _build_agglomerative(n_clusters=k, linkage=linkage, metric=metric)
        labels = model.fit_predict(embeddings)
        centers = compute_cluster_centers(embeddings, labels, noise_label=None)

        meta = {
            "method": "Agglomerative",
            "method_params": {
                "n_clusters": k,
                "linkage": linkage,
                "metric": metric,
            },
            "centers": centers,
        }

    elif method_norm == "dbscan":
        eps = float(params.get("eps", 0.5))
        min_samples = int(params.get("min_samples", 5))
        metric = str(params.get("metric", "euclidean"))

        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = model.fit_predict(embeddings)
        centers = compute_cluster_centers(embeddings, labels, noise_label=-1)

        meta = {
            "method": "DBSCAN",
            "method_params": {
                "eps": eps,
                "min_samples": min_samples,
                "metric": metric,
            },
            "centers": centers,
        }

    else:
        raise ValueError(f"Unsupported clustering method: {method_key}")

    noise_label = int(params.get("noise_label", -1))
    noise_count = int(np.sum(labels == noise_label)) if labels is not None else 0
    total = int(len(labels)) if labels is not None else 0
    n_clusters = len(set(labels.tolist()) - {noise_label}) if labels is not None else 0

    meta.update({
        "runtime_seconds": float(time.time() - start),
        "n_clusters": int(n_clusters),
        "noise_label": noise_label,
        "noise_count": int(noise_count),
        "noise_ratio": float(noise_count / total) if total > 0 else 0.0,
    })

    return labels, meta


def compute_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    noise_label: int = -1,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    metric: str = "euclidean",
    compute_db: bool = True
) -> Dict[str, Any]:
    # Filter noise for metric calculation
    mask = labels != noise_label if labels is not None else None
    if mask is None:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None, "note": "no labels"}

    X = embeddings[mask]
    y = labels[mask]

    if len(set(y.tolist())) < 2:
        return {
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
            "note": "insufficient clusters",
            "n_samples": int(len(y)),
            "n_clusters": int(len(set(y.tolist())))
        }

    X_eval, y_eval = _maybe_sample(X, y, sample_size, random_state)

    out = {
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "n_samples": int(len(y_eval)),
        "n_clusters": int(len(set(y_eval.tolist())))
    }

    try:
        out["silhouette"] = float(silhouette_score(X_eval, y_eval, metric=metric))
    except Exception as e:
        out["silhouette"] = None
        out["silhouette_note"] = str(e)

    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, y_eval))
    except Exception as e:
        out["calinski_harabasz"] = None
        out["calinski_harabasz_note"] = str(e)

    if compute_db:
        try:
            out["davies_bouldin"] = float(davies_bouldin_score(X_eval, y_eval))
        except Exception as e:
            out["davies_bouldin"] = None
            out["davies_bouldin_note"] = str(e)

    return out


def compute_cluster_centers(
    embeddings: np.ndarray,
    labels: np.ndarray,
    noise_label: Optional[int] = None
) -> Dict[int, np.ndarray]:
    centers: Dict[int, np.ndarray] = {}
    for c in sorted(set(labels.tolist())):
        if noise_label is not None and int(c) == int(noise_label):
            continue
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        centers[int(c)] = np.mean(embeddings[idx], axis=0)
    return centers


def _build_agglomerative(n_clusters: int, linkage: str, metric: str):
    # sklearn compatibility across versions
    try:
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)
    except TypeError:
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=metric)


def _maybe_sample(
    X: np.ndarray,
    y: Optional[np.ndarray],
    sample_size: Optional[int],
    random_state: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if sample_size is None or sample_size <= 0:
        return X, y
    n = X.shape[0]
    if n <= sample_size:
        return X, y

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=int(sample_size), replace=False)
    if y is None:
        return X[idx], None
    return X[idx], y[idx]
