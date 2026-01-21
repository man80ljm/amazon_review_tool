# core/plot_k.py
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from core.plot_style import apply_matplotlib_style

@dataclass
class KRecommend:
    best_k: int
    method: str
    score: Optional[float] = None

def _normalize(values: Dict[int, float], invert: bool = False) -> Dict[int, float]:
    vals = [v for v in values.values() if v is not None]
    if not vals:
        return {k: 0.0 for k in values.keys()}
    vmin = min(vals)
    vmax = max(vals)
    if abs(vmax - vmin) < 1e-12:
        return {k: 0.0 for k in values.keys()}
    out = {}
    for k, v in values.items():
        if v is None:
            out[k] = 0.0
            continue
        score = (vmax - v) / (vmax - vmin) if invert else (v - vmin) / (vmax - vmin)
        out[k] = float(score)
    return out

def recommend_k(
    k_to_inertia: Dict[int, float],
    k_to_silhouette: Dict[int, float],
    weight: float = 0.7,
    penalty_threshold: Optional[int] = None,
    penalty_strength: float = 0.0
) -> KRecommend:
    if not k_to_silhouette:
        best_k = max(k_to_inertia, key=lambda k: k_to_inertia[k])
        return KRecommend(best_k=best_k, method="inertia_min")

    w = float(weight)
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0

    ks = sorted(set(k_to_inertia.keys()) & set(k_to_silhouette.keys()))
    if not ks:
        best_k = max(k_to_silhouette, key=lambda k: k_to_silhouette[k])
        return KRecommend(best_k=best_k, method="silhouette_max", score=k_to_silhouette.get(best_k))

    sil_norm = _normalize({k: k_to_silhouette.get(k) for k in ks}, invert=False)
    elbow_norm = _normalize({k: k_to_inertia.get(k) for k in ks}, invert=True)

    best_k = None
    best_score = None
    for k in ks:
        penalty = 0.0
        if penalty_threshold is not None and k > int(penalty_threshold):
            penalty = (k - int(penalty_threshold)) * float(penalty_strength)
        score = w * sil_norm.get(k, 0.0) + (1.0 - w) * elbow_norm.get(k, 0.0) - penalty
        if best_score is None or score > best_score:
            best_score = score
            best_k = k

    return KRecommend(best_k=int(best_k), method="composite", score=best_score)

def plot_k_curves(
    k_to_inertia: Dict[int, float],
    k_to_silhouette: Dict[int, float],
    recommended_k: int = None,
    save_path: str = None,
    lang: str | None = None,
    labels: Dict[str, str] | None = None
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

    labels = labels or {}
    x_label = labels.get("x_label", "K")
    y1_label = labels.get("y1_label", "WCSS / Inertia")
    y2_label = labels.get("y2_label", "Silhouette Score")
    line1_label = labels.get("line1_label", "WCSS/Inertia (Elbow) - solid (blue)")
    line2_label = labels.get("line2_label", "Silhouette Score - dashed (orange)")
    title = labels.get("title", "Optimal K Selection (Elbow & Silhouette)")
    title_with_k = labels.get(
        "title_with_k",
        f"Optimal K Selection (Recommended K={recommended_k})"
    )
    vline_label = labels.get(
        "vline_label",
        f"Recommended K = {recommended_k} - vertical (green)"
    )

    apply_matplotlib_style(lang)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # 1) WCSS / Inertia（实线：蓝色）
    line1, = ax1.plot(
        ks, inertia,
        marker="o",
        linestyle="-",
        linewidth=2,
        color="tab:blue",
        label=line1_label
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 2) Silhouette（虚线：橙色）
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        ks, sil,
        marker="s",
        linestyle="--",
        linewidth=2,
        color="tab:orange",
        label=line2_label
    )
    ax2.set_ylabel(y2_label)

    # 3) 推荐K（竖虚线：绿色）
    vline = None
    if recommended_k is not None:
        vline = ax1.axvline(
            recommended_k,
            linestyle="--",
            linewidth=2,
            color="tab:green",
            label=vline_label
        )
        ax1.set_title(title_with_k)
    else:
        ax1.set_title(title)

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
