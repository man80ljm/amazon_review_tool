# core/keywords.py
from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def top_keywords_by_cluster(texts: List[str], labels: np.ndarray, top_n: int = 12) -> Dict[int, List[str]]:
    out = {}
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    feats = np.array(vectorizer.get_feature_names_out())

    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            out[c] = []
            continue
        # 簇内平均 TF-IDF
        mean_tfidf = X[idx].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        out[c] = feats[top_idx].tolist()
    return out
