"""
BGE-Reranker 精排（CrossEncoder 通路）。

用途：混合召回给出 Top-K 粗排结果后，用 CrossEncoder 精排出最终 Top-N。
原理：
  - 双塔（BGE dense）是「向量各自编码再算相似度」，快但粗；
    CrossEncoder 把 (query, doc) 拼成一个输入过整个模型，交互充分，准但慢。
    所以标准流水线是：双塔粗召回 20 → CrossEncoder 精排 → 取 5。
  - 单例模式：与 embedder 同理，模型常驻显存约 2GB。
  - 4070 8GB：embedder(~2GB) + reranker(~2GB) 同时驻留约 4GB，空闲 6.5GB 够用；
    若与其他大模型共显存，可在 Embedder 侧切 CPU。
"""
from __future__ import annotations

import threading
from pathlib import Path

import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.config import settings


class BGEReranker:
    """BGE-Reranker-v2-m3 单例封装。"""

    _instance: "BGEReranker | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        path = Path(settings.rerank_model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reranker 模型目录不存在: {path}。"
                "请先按 scripts/model_urls_bge-reranker-v2-m3.txt 下载。"
            )
        self._tok = AutoTokenizer.from_pretrained(str(path))
        # 关键：reranker 是 num_labels=1 的序列分类模型，必须用此类加载才有 logits
        self._model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self._model.eval()
        self._device = settings.embed_device if torch.cuda.is_available() else "cpu"
        if settings.use_fp16 and self._device == "cuda":
            self._model = self._model.half()
        self._model = self._model.to(self._device)

    def rerank(self, query: str, docs: list[Document],
               batch_size: int = 8) -> list[tuple[Document, float]]:
        """对候选文档精排。

        返回：[(Document, score)] 按分数降序。分数是原始 logit（无界，只比相对大小）。
        """
        if not docs:
            return []
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                pairs = [(query, d.page_content) for d in batch]
                enc = self._tok(
                    [p[0] for p in pairs], [p[1] for p in pairs],
                    padding=True, truncation=True, max_length=512, return_tensors="pt",
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                logits = self._model(**enc).logits.view(-1)
                scores.extend(logits.float().cpu().tolist())

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return ranked

    @classmethod
    def get(cls) -> "BGEReranker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance


def get_reranker() -> BGEReranker:
    return BGEReranker.get()
