# config.py
from dataclasses import dataclass

@dataclass
class AppConfig:
    required_columns: tuple = ("review_id", "ASIN", "review_text", "Star", "review_time")

    # Step1: 情感模型（英文评论）
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_batch_size: int = 16
    sentiment_max_chars: int = 1200  # 防止超长文本拖慢/报错

    # Step3: Embedding 模型（更适合聚类）
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # Step4: 聚类扫描范围
    k_min: int = 3
    k_max: int = 12
    random_state: int = 42

    # Step5: 每簇输出
    top_keywords: int = 12
    top_representatives: int = 5
