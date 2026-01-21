# core/translate.py
from __future__ import annotations

from typing import List, Optional

import os


class Translator:
    def __init__(self, model_name: str, batch_size: int = 16):
        if not model_name:
            raise ValueError("translator model_name is empty")
        if not os.path.isdir(model_name):
            raise RuntimeError(f"translation model dir not found: {model_name}")

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.model_name = model_name
        self.batch_size = int(batch_size) if batch_size else 16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, local_files_only=True
        )

        tok_max = getattr(self.tokenizer, "model_max_length", None)
        self.max_length = int(tok_max) if tok_max and tok_max > 0 else 512

    def translate(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        def _safe_str(t) -> str:
            if t is None:
                return ""
            return t if isinstance(t, str) else str(t)

        src = [_safe_str(t) for t in texts]
        out = src[:]

        # Indices of non-empty texts
        idxs = [i for i, t in enumerate(src) if t.strip()]
        if not idxs:
            return out

        import torch

        self.model.eval()

        for start in range(0, len(idxs), self.batch_size):
            batch_idxs = idxs[start:start + self.batch_size]
            batch_texts = [src[i] for i in batch_idxs]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=1,
                )

            decoded = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )

            for i, t in zip(batch_idxs, decoded):
                out[i] = t

        return out
