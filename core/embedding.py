import os
import hashlib
import numpy as np
from typing import List, Optional

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        if not model_name:
            raise ValueError("❌ Embedder: model_name 不能为空")
        
        self.model_name = model_name
        
        # 🔥 防御：batch_size 可能是 None
        if batch_size is None or batch_size <= 0:
            print(f"⚠️ batch_size 无效 ({batch_size})，使用默认值 32")
            batch_size = 32
        self.batch_size = int(batch_size)
        
        # 防御：检查模型路径
        if not os.path.isdir(model_name):
            print(f"⚠️ WARNING: {model_name} 不是本地目录，将尝试从 HuggingFace 下载")
        
        try:
            #✅ 关键修复：加载模型时使用 resolve_model_path
            from core.io_utils import resolve_model_path
            real_model_path = resolve_model_path(self.model_name)
            self.model = SentenceTransformer(
                real_model_path,
                device="cpu",  # 打包后建议锁 CPU
                model_kwargs={"local_files_only": True}
            )
        except Exception as e:
            raise RuntimeError(f"❌ 加载 embedding 模型失败: {model_name}\n错误: {e}")

    def _make_cache_key(self, texts: List[str]) -> str:
        """
        根据文本内容 + 模型名 生成稳定 hash
        """
        h = hashlib.sha256()
        h.update(self.model_name.encode("utf-8"))

        # 只取前后文本，避免超大字符串
        if texts:
            h.update(str(len(texts)).encode("utf-8"))
            h.update(str(texts[0][:200]).encode("utf-8"))
            h.update(str(texts[-1][:200]).encode("utf-8"))

        return h.hexdigest()[:16]

    def encode(
        self,
        texts,  # 故意不写类型，兼容各种传入
        cache_path: Optional[str] = None,
        progress=None
    ) -> np.ndarray:
        """
        计算文本 embedding（带缓存）
        
        🔥 终极防御版：无论传入什么垃圾数据都不会崩溃
        """
        
        # ============ 第1层防御：类型检查 ============
        if texts is None:
            raise ValueError(
                "❌ encode(): texts 参数为 None!\n"
                "这通常说明 df_work['_text'] 有问题。\n"
                "请检查数据加载和过滤步骤。"
            )
        
        # 如果是 Series，转列表
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # 如果不是列表，尝试转换
        if not isinstance(texts, list):
            try:
                texts = list(texts)
            except Exception as e:
                raise ValueError(
                    f"❌ encode(): texts 无法转换为列表!\n"
                    f"当前类型: {type(texts)}\n"
                    f"错误: {e}"
                )
        
        # ============ 第2层防御：内容检查 ============
        if len(texts) == 0:
            raise ValueError(
                "❌ encode(): texts 为空列表!\n"
                "这说明过滤后没有数据了。\n"
                "建议：取消'仅负面'或调整过滤策略。"
            )
        
        # ============ 第3层防御：清洗数据 ============
        # 把所有 None/NaN 转成空字符串
        cleaned_texts = []
        for i, t in enumerate(texts):
            if t is None or (isinstance(t, float) and np.isnan(t)):
                cleaned_texts.append("")
            else:
                cleaned_texts.append(str(t).strip())
        
        # 检查是否全是空文本
        non_empty = sum(1 for t in cleaned_texts if t)
        if non_empty == 0:
            raise ValueError(
                f"❌ encode(): {len(cleaned_texts)} 条文本全部为空!\n"
                "请检查数据中的文本列是否有内容。"
            )
        
        print(f"✅ Embedding 数据检查通过: {len(cleaned_texts)} 条, 非空 {non_empty} 条")
        
        # ============ 缓存逻辑 ============
        cache_file = None
        if cache_path:
            os.makedirs(cache_path, exist_ok=True)
            key = self._make_cache_key(cleaned_texts)
            cache_file = os.path.join(
                cache_path,
                f"embeddings_{key}.npy"
            )

            if os.path.exists(cache_file):
                try:
                    emb = np.load(cache_file)
                    if emb.shape[0] == len(cleaned_texts):
                        if progress:
                            progress(len(cleaned_texts), len(cleaned_texts), "✅ Embedding cache loaded")
                        return emb
                    else:
                        print(f"⚠️ 缓存维度不匹配，重新计算")
                except Exception as e:
                    print(f"⚠️ 加载缓存失败: {e}")

        # ============ 真正计算 embedding ============
        if progress:
            progress(0, len(cleaned_texts), "Embedding encoding...")

        try:
            emb = self.model.encode(
                cleaned_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False
            )
        except Exception as e:
            raise RuntimeError(
                f"❌ SentenceTransformer.encode() 失败!\n"
                f"模型: {self.model_name}\n"
                f"文本数: {len(cleaned_texts)}\n"
                f"错误: {e}"
            )

        emb = emb.astype(np.float32)

        # ============ 保存缓存 ============
        if cache_file:
            try:
                np.save(cache_file, emb)
                print(f"✅ 缓存已保存: {cache_file}")
            except Exception as e:
                print(f"⚠️ 保存缓存失败（不影响运行）: {e}")

        if progress:
            progress(len(cleaned_texts), len(cleaned_texts), "✅ Embedding done")

        return emb