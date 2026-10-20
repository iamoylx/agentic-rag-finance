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


def doc_id(d: Document) -> str:
    """稳定的 chunk id：md5(source + 页码 + 正文)。幂等入库的关键。

    公开（原 `_doc_id`）：`HybridRetriever` 需要用它判断「哪些块已经入过库」，
    从而跳过重复编码——那是 CPU 部署下 19 分钟与 0 秒的差别。
    """
    raw = f"{d.metadata.get('source', '')}|{d.metadata.get('page', '')}|{d.page_content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# 兼容旧名（内部与测试均用 doc_id）
_doc_id = doc_id


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
            merged[doc_id(d)] = (d, e)

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

    def present_ids(self, docs: list[Document]) -> set[str]:
        """返回这些 doc 中**已经入库**的 id 集合（只查 id，不取文档/向量）。

        用途：跳过重复编码。Chroma 是持久化的，重启后 9.5k 条向量都还在，
        但旧代码无条件对全部 chunk 重算 dense 向量——GPU 上 ~6s 看不出来，
        CPU 上实测 10014 块要 **19 分钟**（见 evals/bench_cold_start_cpu.py）。

        为什么不用「语料指纹相等就整体跳过」：指纹只能证明**输入没变**，
        证明不了**库里真的有这些向量**（可能上次 upsert 写到一半进程被杀）。
        逐个 id 问一遍是每次启动都真做的核对，代价几秒，换掉 19 分钟的盲信。
        """
        uniq = list(dict.fromkeys(doc_id(d) for d in docs))
        have: set[str] = set()
        batch = 4096   # 与 upsert 同批量：Chroma 单批上限 ~5461
        for i in range(0, len(uniq), batch):
            res = self._col.get(ids=uniq[i:i + batch], include=[])
            have.update(res.get("ids") or [])
        return have

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
