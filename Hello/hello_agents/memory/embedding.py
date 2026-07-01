"""统一嵌入模块（实现 + 提供器）

说明（中文）：
- 提供统一的文本嵌入接口与多实现：本地Transformer、DashScope（通义千问）、TF-IDF兜底。
- 暴露 get_text_embedder()/get_dimension()/refresh_embedder() 供各记忆类型统一使用。
- 通过环境变量优先级：dashscope > local > tfidf。

环境变量（全部由 .env 提供，代码不写死模型名）：
- EMBED_MODEL_TYPE: "dashscope" | "local" | "tfidf"（默认 dashscope）
- EMBED_MODEL_NAME: 模型名称或本地目录路径（如 C:/Files/Models/bge-base-zh-v1.5）
- EMBED_API_KEY:   Embedding API Key（统一命名）
- EMBED_BASE_URL:  Embedding Base URL（统一命名，可选）
"""

from typing import List, Union, Optional
import threading
import os

import numpy as np
from dotenv import load_dotenv

# 可选依赖统一在文件顶部导入；缺失时置为 None，运行到对应后端时再报错
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - 取决于环境是否安装
    SentenceTransformer = None

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except Exception:  # pragma: no cover
    AutoTokenizer = None
    AutoModel = None
    torch = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    TfidfVectorizer = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

load_dotenv()

# bge-zh 系列官方建议：仅对「检索查询」加这个指令前缀，文档/段落不加
_BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


# ==============
# 抽象与实现
# ==============

class EmbeddingModel:
    """嵌入模型基类（最小接口）"""

    def encode(self, texts: Union[str, List[str]]):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class LocalTransformerEmbedding(EmbeddingModel):
    """本地Transformer嵌入

    - 模型名称 / 本地目录路径由调用方（工厂读取 .env）传入，不在此写死默认值。
    - 优先 sentence-transformers，缺失回退 transformers+torch。
    - 向量统一做 L2 归一化（bge 用余弦相似度的前提）。
    - 对 bge 模型的检索查询自动加指令前缀（仅 query，文档不加）。
    """

    def __init__(self, model_name, normalize=True, query_instruction=None):
        self.model_name = model_name
        self.normalize = normalize
        # 仅 bge 系列默认启用查询指令；非 bge 不加。可用 query_instruction 显式覆盖
        if query_instruction is not None:
            self.query_instruction = query_instruction
        elif "bge" in str(model_name).lower():
            self.query_instruction = _BGE_ZH_QUERY_INSTRUCTION
        else:
            self.query_instruction = ""
        self._backend = None  # "st" 或 "hf"
        self._st_model = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._dimension = None
        self._load_backend()

    def _load_backend(self):
        # 优先 sentence-transformers（支持本地目录路径或 HF 模型名）
        if SentenceTransformer is not None:
            try:
                self._st_model = SentenceTransformer(self.model_name)
                test_vec = self._st_model.encode(
                    "test_text", normalize_embeddings=self.normalize
                )
                self._dimension = len(test_vec)
                self._backend = "st"
                return
            except Exception:
                self._st_model = None

        # 回退 transformers + torch（mean pooling）
        if AutoTokenizer is not None and AutoModel is not None and torch is not None:
            try:
                self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._hf_model = AutoModel.from_pretrained(self.model_name)
                self._hf_model.eval()
                with torch.no_grad():
                    inputs = self._hf_tokenizer(
                        "test_text", return_tensors="pt", padding=True, truncation=True
                    )
                    outputs = self._hf_model(**inputs)
                    test_embedding = self._mean_pool(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                    self._dimension = int(test_embedding.shape[1])
                self._backend = "hf"
                return
            except Exception:
                self._hf_tokenizer = None
                self._hf_model = None

        raise ImportError("未找到可用的本地嵌入后端，请安装 sentence-transformers 或 transformers+torch")

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        """按 attention_mask 做 mean pooling（比直接 mean 更准）"""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode(self, texts: Union[str, List[str]], is_query: bool = False):
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        # 仅对检索查询加 bge 指令前缀；入库文档不加
        if is_query and self.query_instruction:
            inputs = [self.query_instruction + t for t in inputs]

        if self._backend == "st":
            vecs = self._st_model.encode(inputs, normalize_embeddings=self.normalize)
            vecs = [v for v in vecs]
        else:
            tokenized = self._hf_tokenizer(
                inputs, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self._hf_model(**tokenized)
                embeddings = self._mean_pool(
                    outputs.last_hidden_state, tokenized["attention_mask"]
                ).cpu().numpy()
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.clip(norms, 1e-12, None)
            vecs = [v for v in embeddings]

        if single:
            return vecs[0]
        return vecs

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)


class TFIDFEmbedding(EmbeddingModel):
    """TF-IDF 简易兜底（在无深度模型时保证可用）"""

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self._vectorizer = None
        self._is_fitted = False
        self._dimension = max_features
        self._init_vectorizer()

    def _init_vectorizer(self):
        if TfidfVectorizer is None:
            raise ImportError("请安装 scikit-learn: pip install scikit-learn")
        self._vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')

    def fit(self, texts: List[str]):
        self._vectorizer.fit(texts)
        self._is_fitted = True
        self._dimension = len(self._vectorizer.get_feature_names_out())

    def encode(self, texts: Union[str, List[str]]):
        if not self._is_fitted:
            raise ValueError("TF-IDF模型未训练，请先调用fit()方法")
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        tfidf_matrix = self._vectorizer.transform(texts)
        embeddings = tfidf_matrix.toarray()
        if single:
            return embeddings[0]
        return [e for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


class DashScopeEmbedding(EmbeddingModel):
    """阿里云 DashScope（通义千问）Embedding / OpenAI兼容REST 模式

    行为：
    - 如提供 base_url，则优先使用 OpenAI 兼容的 REST 接口（POST {base_url}/embeddings）。
    - 否则使用官方 dashscope SDK 的 TextEmbedding.call。
    - 模型名称由调用方（工厂读取 .env）传入，不在此写死默认值。
    """

    def __init__(self, model_name, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._dimension = None
        # 仅在非REST情况下初始化SDK
        if not self.base_url:
            self._init_client()
        # 探测维度
        test = self.encode("health_check")
        self._dimension = len(test)

    def _init_client(self):
        try:
            if self.api_key:
                # 将统一命名的 API Key 注入到 SDK 期望的位置
                os.environ["DASHSCOPE_API_KEY"] = self.api_key
            import dashscope  # noqa: F401
        except ImportError:
            raise ImportError("请安装 dashscope: pip install dashscope")

    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        # REST 模式（OpenAI兼容）
        if self.base_url:
            if requests is None:
                raise ImportError("请安装 requests: pip install requests")
            url = self.base_url.rstrip("/") + "/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                "Content-Type": "application/json",
            }
            payload = {"model": self.model_name, "input": inputs}
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code >= 400:
                raise RuntimeError(f"Embedding REST 调用失败: {resp.status_code} {resp.text}")
            data = resp.json()
            # 期望结构：{"data": [{"embedding": [...]}]}
            items = data.get("data") or []
            vecs = [np.array(item.get("embedding")) for item in items]
            if single:
                return vecs[0]
            return vecs

        # SDK 模式
        from dashscope import TextEmbedding
        rsp = TextEmbedding.call(model=self.model_name, input=inputs)
        embeddings_obj = None
        if isinstance(rsp, dict):
            embeddings_obj = (rsp.get("output") or {}).get("embeddings")
        else:
            embeddings_obj = getattr(getattr(rsp, "output", None), "embeddings", None)
        if not embeddings_obj:
            raise RuntimeError("DashScope 返回为空或格式不匹配")
        vecs = [np.array(item.get("embedding") or item.get("vector")) for item in embeddings_obj]
        if single:
            return vecs[0]
        return vecs

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)


# ==============
# 工厂与回退
# ==============

def create_embedding_model(model_type: str = "local", **kwargs) -> EmbeddingModel:
    """创建嵌入模型实例

    model_type: "dashscope" | "local" | "tfidf"
    kwargs: model_name, api_key, base_url
    """
    if model_type in ("local", "sentence_transformer", "huggingface"):
        return LocalTransformerEmbedding(**kwargs)
    elif model_type == "dashscope":
        return DashScopeEmbedding(**kwargs)
    elif model_type == "tfidf":
        # TF-IDF 不接受 model_name/api_key 等参数，过滤掉
        return TFIDFEmbedding(max_features=int(kwargs.get("max_features", 1000)))
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def create_embedding_model_with_fallback(preferred_type: str = "dashscope", **kwargs) -> EmbeddingModel:
    """带回退的创建：dashscope -> local -> tfidf"""
    if preferred_type in ("sentence_transformer", "huggingface"):
        preferred_type = "local"
    fallback = ["dashscope", "local", "tfidf"]
    # 将首选放最前
    if preferred_type in fallback:
        fallback.remove(preferred_type)
        fallback.insert(0, preferred_type)
    for t in fallback:
        try:
            return create_embedding_model(t, **kwargs)
        except Exception:
            continue
    raise RuntimeError("所有嵌入模型都不可用，请安装依赖或检查配置")


# ==================
# Provider（单例）
# ==================

_lock = threading.RLock()
_embedder: Optional[EmbeddingModel] = None

# 各提供商在 .env 未显式指定 EMBED_MODEL_NAME 时的兜底模型名
_DEFAULT_MODEL_BY_TYPE = {
    "dashscope": "text-embedding-v3",
    "local": "BAAI/bge-base-zh-v1.5",
}


def _build_embedder() -> EmbeddingModel:
    """完全依据 .env 构建嵌入实例（模型名不写死在类里）。"""
    preferred = os.getenv("EMBED_MODEL_TYPE", "dashscope").strip()
    default_model = _DEFAULT_MODEL_BY_TYPE.get(preferred, _DEFAULT_MODEL_BY_TYPE["local"])
    model_name = os.getenv("EMBED_MODEL_NAME", default_model).strip()
    kwargs = {}
    if model_name:
        kwargs["model_name"] = model_name
    api_key = os.getenv("EMBED_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    base_url = os.getenv("EMBED_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return create_embedding_model_with_fallback(preferred_type=preferred, **kwargs)


def get_text_embedder() -> EmbeddingModel:
    """获取全局共享的文本嵌入实例（线程安全单例）"""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            _embedder = _build_embedder()
        return _embedder


def get_dimension(default: int = 384) -> int:
    """获取统一向量维度（失败回退默认值）"""
    try:
        return int(getattr(get_text_embedder(), "dimension", default))
    except Exception:
        return int(default)


def refresh_embedder() -> EmbeddingModel:
    """强制重建嵌入实例（可用于动态切换环境变量）"""
    global _embedder
    with _lock:
        _embedder = _build_embedder()
        return _embedder
