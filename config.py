from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Union
import json
import os

# 用户配置文件名（默认放在项目运行目录）
DEFAULT_SETTINGS_FILE = "settings.json"


def load_user_settings(path: Union[str, os.PathLike, Dict[str, Any], None] = DEFAULT_SETTINGS_FILE) -> Dict[str, Any]:
    """
    读取 settings.json（不存在/不可读则返回空 dict）
    """
    if isinstance(path, dict):
        return {}

    if path is None:
        path = DEFAULT_SETTINGS_FILE

    try:
        if not path or not os.path.isfile(path):
            return {}
    except TypeError:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_user_settings(data: Dict[str, Any], path: str = DEFAULT_SETTINGS_FILE, merge: bool = True) -> None:
    """
    保存 settings.json
    """
    if data is None:
        data = {}

    try:
        if merge:
            old = load_user_settings(path) or {}
            if not isinstance(old, dict):
                old = {}
            old.update(data)
            data_to_write = old
        else:
            data_to_write = data

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_to_write, f, ensure_ascii=False, indent=2)

    except Exception:
        pass


@dataclass
class AppConfig:
    """
    应用全局配置
    """

    # 字段映射
    field_map: Dict[str, str] = field(default_factory=lambda: {
        "text": "ReviewText",
        "asin": "ASIN",
        "star": "Star",
        "time": "ReviewTime"
    })

    # Step1：情感分析
    sentiment_model: Optional[str] = "models/sentiment"
    sentiment_batch_size: int = 16
    sentiment_max_chars: int = 1200

    # 负面筛选策略
    negative_mode: str = "STAR_ONLY"
    star_negative_threshold: float = 4.0
    sentiment_conf_threshold: float = 0.6

    # Step2：Embedding
    embedding_model: str = "models/embedding"
    embedding_batch_size: int = 64  # 🔥 确保有默认值

    # Step3：K 扫描参数
    k_min: int = 2
    k_max: int = 20
    random_state: int = 42

    # Step5：聚类结果分析
    top_keywords: int = 8
    top_representatives: int = 3

    # 文本语言
    text_language: str = "en"

    # 输出
    output_dir: str = "outputs"
    offline_mode: bool = True

    # 报告
    report_title: str = "Review Analysis Report"
    report_subtitle: str = ""
    report_author: str = ""
    report_language: str = "auto"

    def apply_user_settings(self, settings_or_path: Union[Dict[str, Any], str, os.PathLike, None] = DEFAULT_SETTINGS_FILE) -> None:
        """
        从 settings.json 覆盖配置
        🔥 关键防御：确保数值型字段不会变成 None
        """
        if isinstance(settings_or_path, dict):
            data = settings_or_path
        elif settings_or_path is None:
            data = load_user_settings(DEFAULT_SETTINGS_FILE)
        else:
            data = load_user_settings(settings_or_path)

        # 应用覆盖
        for k, v in (data or {}).items():
            if hasattr(self, k):
                setattr(self, k, v)

        # 🔥🔥🔥 关键防御：确保关键字段不为 None
        if not self.embedding_model:
            print("⚠️ embedding_model 为空，恢复默认值")
            self.embedding_model = "models/embedding"
        
        if self.embedding_batch_size is None or self.embedding_batch_size <= 0:
            print(f"⚠️ embedding_batch_size 无效 ({self.embedding_batch_size})，恢复默认值")
            self.embedding_batch_size = 64
        
        if self.sentiment_batch_size is None or self.sentiment_batch_size <= 0:
            print(f"⚠️ sentiment_batch_size 无效 ({self.sentiment_batch_size})，恢复默认值")
            self.sentiment_batch_size = 16
        
        if self.k_min is None or self.k_min <= 0:
            print(f"⚠️ k_min 无效 ({self.k_min})，恢复默认值")
            self.k_min = 2
        
        if self.k_max is None or self.k_max <= 0:
            print(f"⚠️ k_max 无效 ({self.k_max})，恢复默认值")
            self.k_max = 20

        # sentiment_model 允许为空，但不要让它变成非字符串
        if self.sentiment_model is not None and not isinstance(self.sentiment_model, str):
            self.sentiment_model = "models/sentiment"

    def to_dict(self) -> Dict[str, Any]:
        """方便保存配置"""
        return asdict(self)

    def validate_local_models(self) -> None:
        """
        离线模式校验：模型目录存在性
        """
        if not self.offline_mode:
            return

        if not self.embedding_model:
            raise RuntimeError("offline_mode=True 但 embedding_model 未配置")

        if not os.path.isdir(self.embedding_model):
            raise RuntimeError(f"Embedding 模型目录不存在: {self.embedding_model}")

        if self.sentiment_model:
            if not os.path.isdir(self.sentiment_model):
                raise RuntimeError(f"Sentiment 模型目录不存在: {self.sentiment_model}")
            
import os
import sys

def app_base_dir() -> str:
    """
    返回资源根目录：
    - 打包后：dist/ReviewAnalyzer/_internal
    - 开发时：项目当前工作目录（一般就是项目根）
    """
    if getattr(sys, "frozen", False):
        # 你的模型在 _internal 里
        return os.path.join(os.path.dirname(sys.executable), "_internal")
    return os.getcwd()

def resolve_path(p: str) -> str:
    """把相对路径变成基于资源根目录的绝对路径"""
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.join(app_base_dir(), p)
