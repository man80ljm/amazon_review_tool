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

def plot_k_curves(
    k_to_inertia: Dict[int, float],
    k_to_silhouette: Dict[int, float],
    recommended_k: int = None,
    save_path: str = None
):
    """
    绘制 K 选择曲线：
    - 实线（蓝色）：WCSS / Inertia（肘部法）
    - 虚线（橙色）：Silhouette Score（轮廓系数）
    - 竖虚线（绿色）：Recommended K（推荐K）
    """
    ks = sorted(k_to_inertia.keys())
    inertia = [k_to_inertia[k] for k in ks]
    sil = [k_to_silhouette.get(k, np.nan) for k in ks]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # 1) WCSS / Inertia（实线：蓝色）
    line1, = ax1.plot(
        ks, inertia,
        marker="o",
        linestyle="-",
        linewidth=2,
        color="tab:blue",
        label="WCSS/Inertia (Elbow) — solid (blue)"
    )
    ax1.set_xlabel("K")
    ax1.set_ylabel("WCSS / Inertia")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 2) Silhouette（虚线：橙色）
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        ks, sil,
        marker="s",
        linestyle="--",
        linewidth=2,
        color="tab:orange",
        label="Silhouette Score — dashed (orange)"
    )
    ax2.set_ylabel("Silhouette Score")

    # 3) 推荐K（竖虚线：绿色）
    vline = None
    if recommended_k is not None:
        vline = ax1.axvline(
            recommended_k,
            linestyle="--",
            linewidth=2,
            color="tab:green",
            label=f"Recommended K = {recommended_k} — vertical (green)"
        )
        ax1.set_title(f"Optimal K Selection (Recommended K={recommended_k})")
    else:
        ax1.set_title("Optimal K Selection (Elbow & Silhouette)")

    # 合并图例（主轴 + 次轴 + 竖线）
    handles = [line1, line2]
    labels = [line1.get_label(), line2.get_label()]
    if vline is not None:
        handles.append(vline)
        labels.append(vline.get_label())

    ax1.legend(handles, labels, loc="best")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
