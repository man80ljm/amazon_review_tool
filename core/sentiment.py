# core/sentiment.py
import os
from typing import Callable, Optional, List

import numpy as np
import pandas as pd
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

ProgressCb = Optional[Callable[[int, int, str], None]]


class SentimentAnalyzer:
    """
    通用情感分析器（不依赖具体语言）

    设计原则：
    - model_name=None → 关闭情感分析（但 UI 不禁用，只是运行会走兜底）
    - 仅做二分类：positive / negative
    - 作为“过滤步骤”，不是最终结论
    """

    def __init__(
        self,
        model_name: Optional[str],
        batch_size: int = 16,
        max_chars: int = 1200,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars

        if not model_name:
            self.enabled = False
            self.pipe = None
            return

        self.enabled = True
        device = 0 if torch.cuda.is_available() else -1

        # ✅ 关键修复：local_files_only 只能用于 from_pretrained（加载阶段）
        # 不能传进 pipeline(...)，否则会变成运行期参数，最终传给 tokenizer 报错
        local_only = os.path.isdir(model_name)
        pretrained_kwargs = {"local_files_only": True} if local_only else {}
        
        # 加载 tokenizer 和 model
        from core.io_utils import resolve_model_path
        real_model_path = resolve_model_path(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            real_model_path,
            local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            real_model_path,
            local_files_only=True
        )

        # ✅ pipeline 不带 local_files_only
        self.pipe = pipeline(
            task="sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        tokenizer_max = getattr(tokenizer, "model_max_length", None)
        model_max = getattr(model.config, "max_position_embeddings", 512)
        if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 1000000:
            self.max_length = min(tokenizer_max, model_max)
        else:
            self.max_length = model_max

    def _prep(self, text: str) -> str:
        text = (text or "").strip()
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        return text

    def predict(self, texts: List[str], progress: ProgressCb = None, return_conf: bool = False):
        """
        return_conf=False -> 返回 labels: List[str]
        return_conf=True  -> 返回 (labels: List[str], confs: List[float])
        conf 取自 transformers pipeline 输出 dict 的 'score'
        """
        total = len(texts)

        if not self.enabled:
            if progress:
                progress(total, total, "Sentiment disabled")
            labels = ["neutral"] * total
            confs = [0.0] * total
            return (labels, confs) if return_conf else labels

        labels: List[str] = []
        confs: List[float] = []

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = [self._prep(t) for t in texts[start:end]]

            # ✅ 不要传 local_files_only
            try:
                outs = self.pipe(batch, truncation=True, max_length=self.max_length)
            except TypeError:
                # 兼容部分旧 transformers
                outs = self.pipe(batch, truncation=True)

            for o in outs:
                lab_raw = (o.get("label") or "").upper()
                score = o.get("score", None)
                try:
                    score = float(score) if score is not None else 0.0
                except Exception:
                    score = 0.0

                if lab_raw.startswith("POS"):
                    labels.append("positive")
                else:
                    labels.append("negative")

                confs.append(score)

            if progress:
                progress(end, total, f"Sentiment {end}/{total}")

        return (labels, confs) if return_conf else labels

    @staticmethod
    def predict_sentiment_aligned(
        model,
        df: pd.DataFrame,
        text_col: str,
        progress: ProgressCb = None,
        return_conf: bool = False
    ):
        """
        对 df[text_col] 做情感预测，并保证返回与 df 行数一致（按 index 对齐）

        return_conf=False -> 返回 pd.Series(labels)
        return_conf=True  -> 返回 (pd.Series(labels), pd.Series(confs))
        """
        if df is None or len(df) == 0:
            if return_conf:
                return (
                    pd.Series([], dtype="object"),
                    pd.Series([], dtype="float")
                )
            return pd.Series([], dtype="object")

        texts = df[text_col].fillna("").astype(str).tolist()

        if return_conf:
            labels, confs = model.predict(texts, progress=progress, return_conf=True)
            s_lab = pd.Series(labels, index=df.index, dtype="object")
            s_conf = pd.Series(confs, index=df.index, dtype="float")
            return s_lab, s_conf

        preds = model.predict(texts, progress=progress, return_conf=False)
        return pd.Series(preds, index=df.index, dtype="object")
