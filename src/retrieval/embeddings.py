"""
BGE-M3 向量化封装（dense / sparse）。

用途：混合检索的稠密与稀疏向量来源。全项目唯一的 embedding 出口。
原理：
  - BGE-M3 = XLM-RoBERTa 主干 + 两个投影头。
    dense：取 [CLS] 隐状态 L2 归一化，1024 维；
    sparse：sparse_linear.pt 是 (1024→1) 的标量打分层 —— 对每个 token 位置打「词汇重要性分」，
    经 ReLU + log(1+x) 后按 token_id 聚合（同 id 取 max），得到「词项→权重」映射。
  - 用 transformers 手写而非 FlagEmbedding：避免其重依赖链污染 venv（torch 版本风险）。
  - 单例模式：模型加载 ~5s 且常驻约 2GB 显存，全进程共享一份。
⚠️ 显存注意：sparse 投影会产生 (batch, seq_len, 250002) 的中间张量，
   batch=12、seq=1024 时约 6GB —— 所以 sparse 计算 batch 固定为 1，dense 可以大 batch。
"""
from __future__ import annotations

import threading
from pathlib import Path

import torch
from langchain_core.documents import Document
from transformers import AutoModel, AutoTokenizer

from src.core.config import settings


class BGEM3Embedder:
    """BGE-M3 单例封装。通过 get_embedder() 获取，不要直接实例化。"""

    _instance: "BGEM3Embedder | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        path = Path(settings.embed_model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"BGE-M3 模型目录不存在: {path}。"
                "请先按 scripts/model_urls_bge-m3.txt 下载模型到该目录。"
            )

        self._tok = AutoTokenizer.from_pretrained(str(path))
        self._model = AutoModel.from_pretrained(str(path))
        self._model.eval()

        self._device = settings.embed_device if torch.cuda.is_available() else "cpu"
        self._fp16 = settings.use_fp16 and self._device == "cuda"
        if self._fp16:
            self._model = self._model.half()
        self._model = self._model.to(self._device)

        # sparse 投影头：state_dict 含 weight (1,1024) 与 bias (1,) —— 每 token 打一个标量分
        sparse_path = path / "sparse_linear.pt"
        self._sparse_w: torch.Tensor | None = None
        self._sparse_b: torch.Tensor | None = None
        if sparse_path.exists():
            sd = torch.load(str(sparse_path), map_location="cpu", weights_only=True)
            dtype = torch.float16 if self._fp16 else torch.float32
            self._sparse_w = sd["weight"].to(device=self._device, dtype=dtype)
            self._sparse_b = sd["bias"].to(device=self._device, dtype=dtype)

    # ---------- 对外接口 ----------

    def encode_dense(self, texts: list[str], batch_size: int | None = None) -> "torch.Tensor":
        """批量编码 dense 向量。

        用途：文档入库与查询向量化。
        返回：float32 归一化矩阵 (n, 1024)（CPU），归一化后点积即余弦相似度。
        """
        bs = batch_size or settings.embed_batch_size
        outs: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self._tok(
                    texts[i : i + bs], padding=True, truncation=True,
                    max_length=settings.embed_max_length, return_tensors="pt",
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                hidden = self._model(**enc).last_hidden_state
                vecs = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=1)
                outs.append(vecs.float().cpu())
        return torch.cat(outs, dim=0)

    def encode_sparse(self, texts: list[str], max_length: int | None = None) -> list[dict[int, float]]:
        """批量编码 sparse 词项权重。

        用途：学习型稀疏检索（可选通路；BM25 已覆盖稀疏需求时可以不启用）。
        返回：每条文本一个 {token_id: weight} 字典，仅保留权重 > 0 的非特殊 token。
        原理：token 标量分 = log(1 + relu(linear(hidden)))，同 token_id 取 max；
              与 FlagEmbedding 官方实现一致。
        """
        if self._sparse_w is None:
            raise RuntimeError("sparse_linear.pt 缺失，无法计算 sparse 向量")
        ml = max_length or settings.embed_max_length
        special_ids = set(self._tok.all_special_ids)
        results: list[dict[int, float]] = []
        with torch.no_grad():
            for text in texts:
                enc = self._tok(text, padding=False, truncation=True,
                                max_length=ml, return_tensors="pt")
                enc = {k: v.to(self._device) for k, v in enc.items()}
                hidden = self._model(**enc).last_hidden_state[0]        # (seq, 1024)
                scores = hidden @ self._sparse_w.T + self._sparse_b     # (seq, 1)
                scores = torch.log1p(torch.relu(scores)).squeeze(-1)    # (seq,)
                sparse: dict[int, float] = {}
                for tid, val in zip(enc["input_ids"][0].tolist(), scores.tolist()):
                    if tid in special_ids or val <= 0:
                        continue
                    sparse[tid] = max(sparse.get(tid, 0.0), val)
                results.append(sparse)
        return results

    @property
    def device(self) -> str:
        return self._device

    @classmethod
    def get(cls) -> "BGEM3Embedder":
        """获取全局单例。线程安全：模型加载阶段加锁。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double-checked locking
                    cls._instance = cls()
        return cls._instance


def get_embedder() -> BGEM3Embedder:
    return BGEM3Embedder.get()


def embed_documents(docs: list[Document]) -> "torch.Tensor":
    """便捷函数：批量编码 Document 的 page_content。"""
    return get_embedder().encode_dense([d.page_content for d in docs])
