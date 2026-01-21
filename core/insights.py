# core/insights.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from core.plot_style import apply_matplotlib_style

import re

def asin_cluster_percent(df: pd.DataFrame, asin_col="ASIN", cluster_col="cluster_id") -> pd.DataFrame:
    """ASIN × cluster 占比（行归一化 %），自动兼容 cluster 列名"""
    if asin_col not in df.columns:
        raise KeyError(f"缺少列 {asin_col}，当前列：{list(df.columns)}")

    if cluster_col not in df.columns:
        for alt in ("cluster_id", "cluster", "label", "labels"):
            if alt in df.columns:
                cluster_col = alt
                break

    if cluster_col not in df.columns:
        raise KeyError(f"缺少聚类列 cluster_id（或 cluster/label/labels），当前列：{list(df.columns)}")

    pivot = pd.crosstab(df[asin_col], df[cluster_col], normalize="index") * 100
    pivot = pivot.sort_index()
    return pivot

def plot_heatmap(
    pivot_percent: pd.DataFrame,
    save_path: str = None,
    title: str = None,
    lang: str | None = None,
    labels: dict | None = None
):
    """把 ASIN×cluster 的占比画热力图"""
    apply_matplotlib_style(lang)
    labels = labels or {}
    x_label = labels.get("x_label", "Cluster ID")
    y_label = labels.get("y_label", "ASIN")
    title_label = title or labels.get("title", "ASIN × Cluster Distribution (% within ASIN)")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(pivot_percent.values, aspect="auto")
    ax.set_xticks(range(pivot_percent.shape[1]))
    ax.set_xticklabels(pivot_percent.columns.tolist())
    ax.set_yticks(range(pivot_percent.shape[0]))
    ax.set_yticklabels(pivot_percent.index.tolist())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_label)
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

def plot_priority(
    priority_df: pd.DataFrame,
    save_path: str = None,
    lang: str | None = None,
    labels: dict | None = None
):
    """
    🔥 修复：兼容两种列名
    - priority_score（旧版）
    - priority（新版 cluster_priority_safe）
    """
    apply_matplotlib_style(lang)
    labels = labels or {}
    x_label = labels.get("x_label", "Cluster ID")
    y_label = labels.get("y_label", "Priority Score")
    title = labels.get("title", "Cluster Priority Ranking")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # 🔥 自动识别列名
    score_col = None
    for col in ["priority_score", "priority", "score"]:
        if col in priority_df.columns:
            score_col = col
            break
    
    if score_col is None:
        raise KeyError(
            f"priority_df 缺少优先级列！\n"
            f"需要 'priority_score' 或 'priority'，当前列：{list(priority_df.columns)}"
        )
    
    # 🔥 cluster_id 也可能是 cluster
    cid_col = None
    for col in ["cluster_id", "cluster", "cid"]:
        if col in priority_df.columns:
            cid_col = col
            break
    
    if cid_col is None:
        raise KeyError(
            f"priority_df 缺少聚类ID列！\n"
            f"需要 'cluster_id' 或 'cluster'，当前列：{list(priority_df.columns)}"
        )
    
    ax.bar(priority_df[cid_col].astype(str), priority_df[score_col])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig

def cluster_priority_safe(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    star_col: str = "_score",
    group_col: str | None = None
) -> pd.DataFrame:
    """
    安全版 cluster priority：
    - 不依赖 ASIN
    - 所有数值列强制转 numeric
    - 自动跳过 object dtype
    - 不会再出现 mean dtype=object
    """

    work = df.copy()

    # ---------- cluster 列 ----------
    if cluster_col not in work.columns:
        raise ValueError(f"Missing cluster column: {cluster_col}")

    # ---------- 评分列 ----------
    if star_col not in work.columns:
        raise ValueError(f"Missing star column: {star_col}")

    work[star_col] = pd.to_numeric(work[star_col], errors="coerce")

    # ---------- 可选 group ----------
    if group_col and group_col in work.columns:
        group_keys = [group_col, cluster_col]
    else:
        group_keys = [cluster_col]

    # ---------- 聚合 ----------
    agg = (
        work
        .groupby(group_keys, dropna=False)
        .agg(
            review_count=(star_col, "count"),
            mean_star=(star_col, "mean"),
        )
        .reset_index()
    )

    # ---------- priority score ----------
    # 🔥 统一列名：priority（不是 priority_score）
    agg["priority"] = (
        (1 - agg["mean_star"].fillna(0) / 5.0)
        * agg["review_count"]
    )

    agg = agg.sort_values("priority", ascending=False).reset_index(drop=True)
    return agg

def build_attribute_taxonomy(cluster_keywords: dict, topn: int = 3) -> pd.DataFrame:
    """
    cluster_keywords: {cluster_id: ["keyword1","keyword2",...]} 或 {cluster_id: "kw1,kw2,..."}
    输出: cluster_id, attribute_name
    """
    rows = []
    for cid, kws in (cluster_keywords or {}).items():
        if kws is None:
            name = f"Attribute_{cid}"
        elif isinstance(kws, str):
            parts = [p.strip() for p in re.split(r"[,\s、，;；]+", kws) if p.strip()]
            name = " / ".join(parts[:topn]) if parts else f"Attribute_{cid}"
        else:
            parts = [str(x).strip() for x in kws if str(x).strip()]
            name = " / ".join(parts[:topn]) if parts else f"Attribute_{cid}"
        rows.append({"cluster_id": int(cid), "attribute_name": name})
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)

def asin_attribute_share(
    df: pd.DataFrame,
    asin_col: str,
    cluster_col: str,
    taxonomy_df: pd.DataFrame
) -> pd.DataFrame:
    """
    把 ASIN×cluster 的占比 聚合成 ASIN×attribute 的占比
    """
    # 先算 ASIN×cluster %
    pivot_c = asin_cluster_percent(df, asin_col=asin_col, cluster_col=cluster_col)
    # taxonomy: cluster_id -> attribute_name
    m = taxonomy_df.set_index("cluster_id")["attribute_name"].to_dict()

    # 把列名 cluster_id 映射成 attribute_name（多个 cluster 可能映射到同一 attribute，需要 sum）
    tmp = pivot_c.copy()
    tmp.columns = [m.get(int(c), f"Attribute_{c}") for c in tmp.columns]
    # 同名列求和
    tmp = tmp.groupby(axis=1, level=0).sum()
    return tmp

def asin_attribute_pain(
    df: pd.DataFrame,
    asin_col: str,
    cluster_col: str,
    star_col: str,
    taxonomy_df: pd.DataFrame
) -> pd.DataFrame:
    """
    pain：在每个 ASIN 内，对每个 cluster 计算 priority，再映射到 attribute 并聚合
    输出 ASIN×attribute 的 pain 值（越大越痛）
    """
    # cluster priority by ASIN（返回列：asin, cluster_id, priority, review_count, mean_star...）
    pr = cluster_priority_safe(df, cluster_col=cluster_col, star_col=star_col, group_col=asin_col)

    m = taxonomy_df.set_index("cluster_id")["attribute_name"].to_dict()
    pr["attribute_name"] = pr["cluster_id"].map(lambda x: m.get(int(x), f"Attribute_{x}"))

    # ASIN×attribute 聚合 priority（sum 最直观：越多/越严重累加越大）
    agg = pr.groupby([asin_col, "attribute_name"], as_index=False)["priority"].sum()

    pivot = agg.pivot(index=asin_col, columns="attribute_name", values="priority").fillna(0.0)

    # 排序：列按整体痛点降序
    col_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot[col_order]
    return pivot

def opportunity_insights(
    pain_pivot: pd.DataFrame,
    topk: int = 10
) -> pd.DataFrame:
    """
    输入：ASIN×attribute pain
    输出：机会点表（asin, attribute, pain, baseline_mean, delta）
    """
    baseline = pain_pivot.mean(axis=0)  # 每个 attribute 的全品类均值
    rows = []
    for asin in pain_pivot.index:
        row = pain_pivot.loc[asin]
        delta = row - baseline
        # 只取 delta>0 的机会点（比行业均值更痛）
        for attr, d in delta.sort_values(ascending=False).items():
            if d <= 0:
                continue
            rows.append({
                "asin": asin,
                "attribute": attr,
                "pain": float(row[attr]),
                "baseline_mean": float(baseline[attr]),
                "delta": float(d)
            })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["delta", "pain"], ascending=[False, False]).head(topk).reset_index(drop=True)
