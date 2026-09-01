"""
混合检索器：BM25 + dense 向量 双路召回 → RRF 融合 → Reranker 精排。

用途：P1-S3 的主检索入口。graph 层只调 HybridRetriever.search()，不感知内部细节。
原理：
  - 双路互补：dense 管「同义表述」，BM25 管「精确术语与数字」（证券代码、日期、科目名）。
  - 融合用 RRF 而非加权求和：向量余弦与 BM25 分数量纲完全不同，
    加权需要调参且不稳；RRF 只用排名（score = Σ 1/(k + rank_i)），免归一化。
    k=60 来自 Cormack et al., SIGIR 2009 的原论文设定。
  - 精排兜底：CrossEncoder 对 (query, doc) 充分交互，把粗排顺序纠正为精排顺序。
"""
from __future__ import annotations

from langchain_core.documents import Document

from src.core.config import settings
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import BGEM3Embedder
from src.retrieval.reranker import BGEReranker
from src.retrieval.vector_store import ChromaVectorStore


def rrf_fuse(*ranked_lists: list[Document], k: int | None = None,
             weights: list[float] | None = None) -> list[tuple[Document, float]]:
    """RRF 融合多路召回结果（支持各路加权）。

    参数：每一路都是已按相关性排序的 Document 列表（只看名次，不看原始分数）；
         weights 与 ranked_lists 一一对应，None 则等权。
    返回：[(Document, rrf_score)] 融合分降序。
    原理：score(d) = Σ_over_lists w_l * 1/(k + rank_i(d))，未出现的路不计项。
         加权的动机：财报语料里 BM25 命中的「科目名词面匹配块」远多于真正答案块，
         等权时其长尾排名会把 dense 高排名挤掉（消融数据见 METRICS.md）。
    """
    k = k if k is not None else settings.rrf_k
    weights = weights or [1.0] * len(ranked_lists)
    assert len(weights) == len(ranked_lists), "权重数与召回路数不一致"
    scores: dict[str, float] = {}
    first_seen: dict[str, Document] = {}

    for w, lst in zip(weights, ranked_lists):
        for rank, doc in enumerate(lst, start=1):
            key = f"{doc.metadata.get('source')}|{doc.metadata.get('page')}|{doc.page_content}"
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
            if key not in first_seen:
                first_seen[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(first_seen[key], s) for key, s in ranked]


class HybridRetriever:
    """混合检索器。用途：graph 层的唯一检索入口。"""

    def __init__(
        self,
        docs: list[Document],
        use_rerank: bool = True,
        embedder: BGEM3Embedder | None = None,
        reranker: BGEReranker | None = None,
    ):
        """构建索引并就绪。

        参数：docs 为分块后的 Document 全集（建 BM25 索引 + 入 Chroma）。
              已入库时 Chroma 会幂等覆盖，重复构建无副作用。
        """
        self.docs = list(docs)

        # 路 1：BM25（内存索引）
        self.bm25 = BM25Retriever().build(self.docs)

        # 路 2：dense 向量（Chroma 持久化）
        self.embedder = embedder or BGEM3Embedder.get()
        self.store = ChromaVectorStore()
        embeddings = self.embedder.encode_dense([d.page_content for d in self.docs])
        self.store.upsert(self.docs, embeddings)

        # 精排（可选：显存紧张或只想测召回时可关）
        self._use_rerank = use_rerank
        self.reranker = reranker if (reranker or not use_rerank) else BGEReranker.get()

    # ---------- 检索 ----------

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Document, float]]:
        """完整检索流水线：双路召回 → RRF → 精排。

        返回：[(Document, score)]。精排开启时 score 为 rerank logit，否则为 RRF 分。
        """
        k_final = top_k or settings.top_k_final

        # 双路各取 Top-N 粗召回（调用顺序：BM25 在前、dense 在后，与权重一一对应）
        bm25_hits = self.bm25.search(query, top_k=settings.top_k_retrieve)
        dense_hits = self.store.search_dense(
            self.embedder.encode_dense([query])[0], top_k=settings.top_k_retrieve
        )

        fused = rrf_fuse(
            [d for d, _ in bm25_hits], [d for d, _ in dense_hits],
            weights=[settings.rrf_bm25_weight, settings.rrf_dense_weight],
        )
        candidates = [d for d, _ in fused[: settings.top_k_retrieve]]

        if self._use_rerank and self.reranker is not None:
            return self.reranker.rerank(query, candidates)[:k_final]
        return candidates[:k_final]

    # ---------- 评测钩子（P1-S3 三组对比实验用）----------

    def search_dense_only(self, query: str, top_k: int | None = None) -> list[Document]:
        hits = self.store.search_dense(
            self.embedder.encode_dense([query])[0], top_k=top_k
        )
        return [d for d, _ in hits]

    def search_bm25_only(self, query: str, top_k: int | None = None) -> list[Document]:
        return [d for d, _ in self.bm25.search(query, top_k=top_k)]

    def search_hybrid_no_rerank(self, query: str, top_k: int | None = None) -> list[Document]:
        bm25_hits = self.bm25.search(query, top_k=settings.top_k_retrieve)
        dense_hits = self.store.search_dense(
            self.embedder.encode_dense([query])[0], top_k=settings.top_k_retrieve
        )
        fused = rrf_fuse(
            [d for d, _ in bm25_hits], [d for d, _ in dense_hits],
            weights=[settings.rrf_bm25_weight, settings.rrf_dense_weight],
        )
        return [d for d, _ in fused[:top_k or settings.top_k_final]]
