import numpy as np

from core.clustering import run_clustering, compute_cluster_metrics


def main():
    rng = np.random.default_rng(42)

    # Create three compact blobs
    c1 = rng.normal(loc=0.0, scale=0.4, size=(150, 8))
    c2 = rng.normal(loc=3.0, scale=0.5, size=(150, 8))
    c3 = rng.normal(loc=-3.0, scale=0.6, size=(150, 8))
    X = np.vstack([c1, c2, c3])

    configs = {
        "KMeans": {"n_clusters": 3, "init": "k-means++", "n_init": "auto", "max_iter": 300, "tol": 1e-4},
        "Agglomerative": {"n_clusters": 3, "linkage": "ward", "metric": "euclidean"},
        "DBSCAN": {"eps": 1.2, "min_samples": 5, "metric": "euclidean"},
    }

    for method, params in configs.items():
        labels, meta = run_clustering(X, method=method, params=params, random_state=42)
        metrics = compute_cluster_metrics(X, labels, noise_label=-1, sample_size=2000, random_state=42)
        print("===", method, "===")
        print("n_clusters:", meta.get("n_clusters"), "noise_ratio:", meta.get("noise_ratio"))
        print("metrics:", metrics)


if __name__ == "__main__":
    main()
