# core/plot_k.py
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class KRecommend:
    best_k: int
    method: str

def recommend_k(k_to_silhouette: Dict[int, float]) -> KRecommend:
    best_k = max(k_to_silhouette, key=lambda k: k_to_silhouette[k])
    return KRecommend(best_k=best_k, method="silhouette_max")

def plot_k_curves(k_to_inertia: Dict[int, float],
                  k_to_silhouette: Dict[int, float],
                  recommended_k: int = None,
                  save_path: str = None):
    ks = sorted(k_to_inertia.keys())
    inertia = [k_to_inertia[k] for k in ks]
    sil = [k_to_silhouette.get(k, np.nan) for k in ks]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ks, inertia, marker="o")
    ax1.set_xlabel("K")
    ax1.set_ylabel("WCSS / Inertia")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ks, sil, marker="s", linestyle="--")
    ax2.set_ylabel("Silhouette Score")

    if recommended_k is not None:
        ax1.axvline(recommended_k, linestyle="--")
        ax1.set_title(f"Elbow & Silhouette (Recommended K={recommended_k})")
    else:
        ax1.set_title("Elbow & Silhouette")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
