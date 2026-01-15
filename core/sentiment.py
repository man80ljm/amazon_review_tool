# core/sentiment.py
from typing import Callable, Optional, List
import torch
from transformers import pipeline

ProgressCb = Optional[Callable[[int, int, str], None]]

class SentimentAnalyzer:
    def __init__(self, model_name: str, batch_size: int = 16, max_chars: int = 1200):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars

        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("sentiment-analysis", model=model_name, device=device)

    def _prep(self, text: str) -> str:
        text = (text or "").strip()
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        return text

    def predict(self, texts: List[str], progress: ProgressCb = None) -> List[str]:
        labels = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = [self._prep(t) for t in texts[start:end]]
            try:
                outs = self.pipe(batch, truncation=True)
            except TypeError:
                # 兼容老版本 transformers
                outs = self.pipe(batch)

            for o in outs:
                lab = o.get("label", "")
                # SST-2: POSITIVE/NEGATIVE
                if lab.upper().startswith("POS"):
                    labels.append("positive")
                else:
                    labels.append("negative")

            if progress:
                progress(end, total, f"Sentiment {end}/{total}")
        return labels
