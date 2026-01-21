# core/keywords.py
from typing import Dict, List
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# 中文分词（jieba）
# -------------------------
import jieba

_ZH_STOPWORDS = {
    "的", "了", "和", "是", "就", "都", "而", "及", "与", "着", "或", "也", "很",
    "在", "有", "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这个", "那个", "这些", "那些", "一个", "一样", "非常", "比较",
    "不是", "没有", "感觉", "觉得", "还是", "真的",
}

def _clean_zh(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text

def _zh_tokenize(text: str) -> List[str]:
    """
    中文分词（jieba）
    仅用于 cluster 关键词解释，不参与 embedding
    """
    text = _clean_zh(text)
    if not text:
        return []

    tokens = jieba.lcut(text)

    out: List[str] = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t in _ZH_STOPWORDS:
            continue
        if len(t) <= 1:
            continue
        if t.isdigit():
            continue
        out.append(t)

    return out


# -------------------------
# 关键词提取（按 cluster）
# -------------------------
def top_keywords_by_cluster(
    texts: List[str],
    labels: np.ndarray,
    top_n: int = 12,
    language: str = "en"
) -> Dict[int, List[str]]:
    """
    为每个 cluster 提取 Top-N TF-IDF 关键词

    说明：
    - 英文：TF-IDF + stop_words
    - 中文：jieba 分词 + TF-IDF
    - 这是“解释层”，不是决定聚类的核心
    """

    out: Dict[int, List[str]] = {}

    def _safe_text(t) -> str:
        if isinstance(t, str):
            return t
        if t is None:
            return ""
        try:
            if isinstance(t, float) and np.isnan(t):
                return ""
            if np.isnan(t):
                return ""
        except Exception:
            pass
        return str(t)

    texts = [_safe_text(t) for t in texts]
    if not any(t.strip() for t in texts):
        for c in sorted(set(labels.tolist())):
            out[c] = []
        return out

    if language.lower().startswith("zh"):
        vectorizer = TfidfVectorizer(
            max_features=8000,
            tokenizer=_zh_tokenize,
            token_pattern=None,      # 使用 tokenizer 时必须关闭正则
            ngram_range=(1, 2)
        )
    else:
        vectorizer = TfidfVectorizer(
            max_features=6000,
            stop_words="english",
            ngram_range=(1, 2)
        )

    X = vectorizer.fit_transform(texts)
    features = np.array(vectorizer.get_feature_names_out())

    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            out[c] = []
            continue

        mean_tfidf = X[idx].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        out[c] = features[top_idx].tolist()

    return out
