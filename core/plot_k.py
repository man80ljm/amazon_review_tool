# core/plot_k.py
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class KRecommend:
    best_k: int
    method: str

def _recommend_by_elbow(k_to_inertia: Dict[int, float]) -> int:
    ks = sorted(k_to_inertia.keys())
    if len(ks) < 3:
        return ks[0]
    inertia = np.array([k_to_inertia[k] for k in ks], dtype=float)
    inertia = np.clip(inertia, 1e-9, None)
    log_inertia = np.log(inertia)
    curvatures = []
    for i in range(1, len(ks) - 1):
        curvature = abs(log_inertia[i - 1] - 2 * log_inertia[i] + log_inertia[i + 1])
        curvatures.append(curvature)
    best_idx = int(np.argmax(curvatures)) + 1
    return ks[best_idx]

def _normalize_scores(scores: Dict[int, float], higher_better: bool) -> Dict[int, float]:
    values = np.array([v for v in scores.values() if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return {k: 0.0 for k in scores.keys()}
    v_min = float(values.min())
    v_max = float(values.max())
    if np.isclose(v_min, v_max):
        return {k: 0.5 for k in scores.keys()}
    norm = {}
    for k, v in scores.items():
        if not np.isfinite(v):
            norm[k] = 0.0
            continue
        scaled = (v - v_min) / (v_max - v_min)
        norm[k] = float(scaled if higher_better else (1.0 - scaled))
    return norm

def recommend_k(
    k_to_inertia: Dict[int, float],
    k_to_silhouette: Dict[int, float],
    k_to_calinski_harabasz: Dict[int, float],
    k_to_davies_bouldin: Dict[int, float],
) -> KRecommend:
    candidates = {}
    if k_to_silhouette:
        candidates["silhouette_max"] = max(k_to_silhouette, key=lambda k: k_to_silhouette[k])
    if k_to_calinski_harabasz:
        candidates["calinski_harabasz_max"] = max(k_to_calinski_harabasz, key=lambda k: k_to_calinski_harabasz[k])
    if k_to_davies_bouldin:
        candidates["davies_bouldin_min"] = min(k_to_davies_bouldin, key=lambda k: k_to_davies_bouldin[k])
    if k_to_inertia:
        candidates["elbow_curvature"] = _recommend_by_elbow(k_to_inertia)

    vote_counts = {}
    for k in candidates.values():
        vote_counts[k] = vote_counts.get(k, 0) + 1

    sil_norm = _normalize_scores(k_to_silhouette, higher_better=True)
    ch_norm = _normalize_scores(k_to_calinski_harabasz, higher_better=True)
    db_norm = _normalize_scores(k_to_davies_bouldin, higher_better=False)

    def composite_score(k: int) -> float:
        return sil_norm.get(k, 0.0) + ch_norm.get(k, 0.0) + db_norm.get(k, 0.0)

    if vote_counts:
        max_votes = max(vote_counts.values())
        top_k = [k for k, v in vote_counts.items() if v == max_votes]
        best_k = max(top_k, key=composite_score)
        method = "vote:" + ",".join(sorted([m for m, k in candidates.items() if k == best_k]))
    else:
        best_k = max(k_to_silhouette, key=lambda k: k_to_silhouette[k])
        method = "silhouette_max"

    return KRecommend(best_k=best_k, method=method)

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
