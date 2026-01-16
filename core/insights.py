# core/insights.py
import os
import pandas as pd
import matplotlib.pyplot as plt

def asin_cluster_percent(df: pd.DataFrame, asin_col="ASIN", cluster_col="cluster_id") -> pd.DataFrame:
    """ASIN × cluster 占比（行归一化 %）"""
    pivot = pd.crosstab(df[asin_col], df[cluster_col], normalize="index") * 100
    pivot = pivot.sort_index()
    return pivot

def plot_heatmap(pivot_percent: pd.DataFrame, save_path: str = None, title: str = None):
    """把 ASIN×cluster 的占比画热力图"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(pivot_percent.values, aspect="auto")
    ax.set_xticks(range(pivot_percent.shape[1]))
    ax.set_xticklabels(pivot_percent.columns.tolist())
    ax.set_yticks(range(pivot_percent.shape[0]))
    ax.set_yticklabels(pivot_percent.index.tolist())
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("ASIN")
    ax.set_title(title or "ASIN × Cluster Distribution (% within ASIN)")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig

def cluster_priority(df: pd.DataFrame, cluster_col="cluster_id", star_col="Star") -> pd.DataFrame:
    """
    简单稳妥的优先级：
    ratio × (5 - mean_star)
    """
    total = len(df)
    g = df.groupby(cluster_col)
    out = pd.DataFrame({
        "cluster_id": g.size().index,
        "cluster_size": g.size().values,
        "ratio": (g.size() / total).values,
        "mean_star": g[star_col].mean().values
    })
    out["severity"] = 5 - out["mean_star"]
    out["priority_score"] = out["ratio"] * out["severity"]
    out = out.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return out

def plot_priority(priority_df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(priority_df["cluster_id"].astype(str), priority_df["priority_score"])
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Priority Score")
    ax.set_title("Cluster Priority (ratio × (5 - mean_star))")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
