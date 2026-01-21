import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def download_sentence_transformer(model_id: str, out_dir: str) -> None:
    """Download and save a sentence-transformers model for offline usage."""
    from sentence_transformers import SentenceTransformer

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Embedding] Downloading ST model: {model_id}")
    model = SentenceTransformer(model_id)
    model.save(out_dir)
    print(f"[Embedding] Saved to: {out_dir}")


def download_hf_sentiment(model_id: str, out_dir: str) -> None:
    """
    Download a HF Transformers sentiment model to local dir:
    - tokenizer
    - model
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Sentiment] Downloading HF model: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)

    tok.save_pretrained(out_dir)
    mdl.save_pretrained(out_dir)
    print(f"[Sentiment] Saved to: {out_dir}")


def download_hf_translation(model_id: str, out_dir: str) -> None:
    """
    Download a HF Transformers seq2seq model to local dir:
    - tokenizer
    - model
    """
    try:
        import sentencepiece  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: sentencepiece. "
            "Install it with: pip install sentencepiece"
        ) from e

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Translate] Downloading HF model: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    tok.save_pretrained(out_dir)
    mdl.save_pretrained(out_dir)
    print(f"[Translate] Saved to: {out_dir}")


# Sentiment model catalog. Add more entries as needed.
# Key -> {id, lang}
SENTIMENT_CATALOG: Dict[str, Dict[str, str]] = {
    # English (SST-2)
    "en_sst2": {"id": "distilbert-base-uncased-finetuned-sst-2-english", "lang": "en"},
    # Chinese (JD reviews)
    "zh_jd_binary": {"id": "uer/roberta-base-finetuned-jd-binary-chinese", "lang": "zh"},
    # Chinese (Dianping reviews)
    "zh_dianping": {"id": "uer/roberta-base-finetuned-dianping-chinese", "lang": "zh"},
    # Chinese (General fallback)
    "zh_general": {"id": "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment", "lang": "zh"},
    # Chinese (News sentiment, alternative fallback)
    "zh_chinanews": {"id": "uer/roberta-base-finetuned-chinanews-chinese", "lang": "zh"},
}

# Translation model catalog. Key -> {id, pair}
TRANSLATE_CATALOG: Dict[str, Dict[str, str]] = {
    "zh_en": {"id": "Helsinki-NLP/opus-mt-zh-en", "pair": "zh_en"},
    "en_zh": {"id": "Helsinki-NLP/opus-mt-en-zh", "pair": "en_zh"},
}


def _safe_dirname(name: str) -> str:
    return name.replace("/", "--").replace("\\", "--").replace(":", "--")


def _parse_kv_list(items: Iterable[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid format (expected key=id): {item}")
        key, model_id = item.split("=", 1)
        key = key.strip()
        model_id = model_id.strip()
        if not key or not model_id:
            raise ValueError(f"Invalid key/id in: {item}")
        pairs.append((key, model_id))
    return pairs


def resolve_sentiment_models(selection: str, extra_models: Iterable[str]) -> List[Tuple[str, str]]:
    resolved: List[Tuple[str, str]] = []

    if selection not in {"en", "zh", "all"}:
        raise ValueError(f"Unsupported sentiment set: {selection}")

    for key, meta in SENTIMENT_CATALOG.items():
        if selection == "all" or meta["lang"] == selection:
            resolved.append((key, meta["id"]))

    resolved.extend(_parse_kv_list(extra_models))
    return resolved


def resolve_translate_models(selection: str, extra_models: Iterable[str]) -> List[Tuple[str, str]]:
    resolved: List[Tuple[str, str]] = []

    if selection not in {"none", "zh_en", "en_zh", "all"}:
        raise ValueError(f"Unsupported translate set: {selection}")

    if selection != "none":
        for key, meta in TRANSLATE_CATALOG.items():
            if selection == "all" or meta["pair"] == selection:
                resolved.append((key, meta["id"]))

    resolved.extend(_parse_kv_list(extra_models))
    return resolved


def _should_skip_download(out_dir: str, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    if not os.path.isdir(out_dir):
        return False
    return any(os.scandir(out_dir))


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download embedding + sentiment models for offline usage.",
    )
    parser.add_argument(
        "--embedding-id",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Override embedding model id.",
    )
    parser.add_argument(
        "--sentiment-set",
        choices=["en", "zh", "all"],
        default="all",
        help="Which sentiment model set to download (default: all).",
    )
    parser.add_argument(
        "--sentiment-model",
        action="append",
        default=[],
        help="Extra sentiment model in key=id format. Can be repeated.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download when target folder already has files.",
    )
    parser.add_argument(
        "--translate-set",
        choices=["none", "zh_en", "en_zh", "all"],
        default="none",
        help="Which translation model set to download (default: none).",
    )
    parser.add_argument(
        "--translate-model",
        action="append",
        default=[],
        help="Extra translation model in key=id format. Can be repeated.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    emb_dir = project_root / "models" / "embedding"
    sent_root = project_root / "models" / "sentiment"
    trans_root = project_root / "models" / "translate"

    embedding_id = args.embedding_id
    sentiment_models = resolve_sentiment_models(args.sentiment_set, args.sentiment_model)
    translate_models = resolve_translate_models(args.translate_set, args.translate_model)

    if _should_skip_download(str(emb_dir), args.skip_existing):
        print(f"[Embedding] Skipped (exists): {emb_dir}")
    else:
        download_sentence_transformer(embedding_id, str(emb_dir))
    for key, model_id in sentiment_models:
        out_dir = sent_root / _safe_dirname(key)
        if _should_skip_download(str(out_dir), args.skip_existing):
            print(f"[Sentiment] Skipped (exists): {out_dir}")
            continue
        download_hf_sentiment(model_id, str(out_dir))

    for key, model_id in translate_models:
        out_dir = trans_root / _safe_dirname(key)
        if _should_skip_download(str(out_dir), args.skip_existing):
            print(f"[Translate] Skipped (exists): {out_dir}")
            continue
        download_hf_translation(model_id, str(out_dir))

    print("\nAll done.")
    print(f"Embedding dir : {emb_dir}")
    print(f"Sentiment root: {sent_root}")
    print(f"Translate root: {trans_root}")


if __name__ == "__main__":
    main()
