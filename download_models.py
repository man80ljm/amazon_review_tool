import os
from pathlib import Path


def download_sentence_transformer(model_id: str, out_dir: str):
    """下载并用 sentence-transformers 的 save() 落地（保证目录结构可被 SentenceTransformer 直接加载）"""
    from sentence_transformers import SentenceTransformer

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Embedding] Downloading ST model: {model_id}")
    model = SentenceTransformer(model_id)   # 需要联网
    model.save(out_dir)
    print(f"[Embedding] Saved to: {out_dir}")


def download_hf_sentiment(model_id: str, out_dir: str):
    """
    下载 HuggingFace Transformers 的情感模型并落地到本地目录：
    - 保存 tokenizer
    - 保存 model
    之后你可以用 from_pretrained(out_dir) 离线加载
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


def main():
    project_root = Path(__file__).resolve().parent
    emb_dir = project_root / "models" / "embedding"
    sent_dir = project_root / "models" / "sentiment"

    # ✅ 推荐：中英都能用的 embedding（你现在分析中文点评，必须用多语种 embedding 才合理）
    embedding_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ✅ 情感模型：先给你一个“英文情感”稳定的（你如果要中文情感我再给你换）
    # 英文推荐：
    sentiment_id = "distilbert-base-uncased-finetuned-sst-2-english"

    download_sentence_transformer(embedding_id, str(emb_dir))
    download_hf_sentiment(sentiment_id, str(sent_dir))

    print("\n✅ All done.")
    print(f"Embedding dir : {emb_dir}")
    print(f"Sentiment dir : {sent_dir}")


if __name__ == "__main__":
    main()
