"""
Chroma 向量库封装（dense 通路持久化存储）。

用途：dense 向量的存取。开发期用 Chroma 本地持久化；生产可切 Milvus ——
     本类即 VectorStore 抽象的 Chroma 实现，切换时只需新增一个同接口实现类。
原理：
  - 向量由外部（BGEM3Embedder）算好直接传入，不依赖 Chroma 的 embedding_function，
    保证「算向量」与「存向量」解耦，方便以后换模型只重建索引、不改存储层。
  - id 用 page_content 的 md5：同一 chunk 重复入库时自动覆盖（幂等），避免翻倍。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from langchain_core.documents import Document

from src.core.config import settings


def _doc_id(d: Document) -> str:
    """稳定的 chunk id：md5(正文 + 页码)。幂等入库的关键。"""
    raw = f"{d.metadata.get('source', '')}|{d.metadata.get('page', '')}|{d.page_content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class ChromaVectorStore:
    """Chroma 持久化向量库。"""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "finance_reports",
    ):
        persist = str(persist_dir or settings.chroma_dir)
        Path(persist).mkdir(parents=True, exist_ok=True)
        # PersistentClient：数据落盘项目内 data/chroma/，进程重启不丢
        self._client = chromadb.PersistentClient(path=persist)
        # cosine space：BGE 向量已归一化，cosine 与点积等价，但显式声明更不易出错
        self._col = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ---------- 写 ----------

    def upsert(self, docs: list[Document], embeddings) -> int:
        """幂等写入（id 相同即覆盖）。

        参数：embeddings 为 torch.Tensor (n, 1024) 或等价 list，来自 BGEM3Embedder。
        返回：本次处理的文档数。
        """
        if not docs:
            return 0
        if hasattr(embeddings, "tolist"):  # torch.Tensor / numpy
            embeddings = embeddings.tolist()
        assert len(docs) == len(embeddings), "文档数与向量数不一致"

        # 去重：真实年报存在「同页同内容」的重复块（重复段落/相同表格），
        # md5(source|page|content) 会撞 id；chroma upsert 要求 id 唯一，
        # 这里按 id 合并、保留最后一次出现（与「幂等覆盖」语义一致）
        merged: dict[str, tuple] = {}
        for d, e in zip(docs, embeddings):
            merged[_doc_id(d)] = (d, e)

        # chroma 有单批上限（1.5.x 为 max_batch_size≈5461），大语料必须分批
        pairs = list(merged.items())
        batch = 4096
        for i in range(0, len(pairs), batch):
            part = pairs[i:i + batch]
            self._col.upsert(
                ids=[k for k, _ in part],
                documents=[p[0].page_content for _, p in part],
                metadatas=[self._flat_meta(p[0].metadata) for _, p in part],
                embeddings=[p[1] for _, p in part],
            )
        return len(merged)

    # ---------- 读 ----------

    def search_dense(self, query_embedding, top_k: int | None = None,
                     where: dict | None = None) -> list[tuple[Document, float]]:
        """dense 检索。

        返回：[(Document, similarity)]，相似度降序。
        原理：Chroma cosine distance ∈ [0,2]，similarity = 1 - distance（归一化向量下即余弦值）。
        """
        k = top_k or settings.top_k_retrieve
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
        res = self._col.query(
            query_embeddings=[query_embedding],  # type: ignore[list-item]
            n_results=min(k, max(self._col.count(), 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[tuple[Document, float]] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append((Document(page_content=doc, metadata=dict(meta)), 1.0 - float(dist)))
        return out

    # ---------- 其他 ----------

    def count(self) -> int:
        return self._col.count()

    @staticmethod
    def _flat_meta(meta: dict) -> dict:
        """Chroma metadata 只接受标量值，list/dict 需序列化成字符串。"""
        out = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
        return out
