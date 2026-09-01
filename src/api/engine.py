"""
服务层引擎单例：模型 / 检索器 / 图的懒加载与重建。

语法：模块级字典 + threading.Lock，FastAPI lifespan 里调用 startup()。
原理：
  - BGE-M3 / Reranker 加载约 5-6 秒，绝不能每个请求加载一次 → 进程内单例。
  - /api/ingest 入库新文档后需要重建检索器与图（retrieve 节点是闭包持有检索器），
    重建期间用锁挡住并发请求；checkpointer 复用，多轮会话不丢。
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver

from src.core.config import settings
from src.graph.rag_graph import build_rag_graph
from src.ingestion.chunker import FinancialChunker
from src.ingestion.pdf_parser import AnnualReportParser
from src.retrieval.hybrid import HybridRetriever


class Engine:
    """持有 retriever / graph / checkpointer 的全局引擎。"""

    def __init__(self) -> None:
        self.retriever: HybridRetriever | None = None
        self.graph = None
        self.checkpointer = MemorySaver()
        self.lock = threading.Lock()   # ingest 重建时挡并发
        self.ready = False

    # ---------- 启动：解析 data/raw 全部 PDF 并建索引 ----------
    def startup(self) -> int:
        """冷启动建库。返回入库分块数。幂等：Chroma upsert + BM25 重建都无副作用。"""
        settings.ensure_dirs()
        docs, _ = self.build_corpus()
        self._build(docs)
        self.ready = True
        return len(docs)

    # ---------- 语料构建（带 fingerprint 缓存）----------
    def build_corpus(self) -> tuple[list[Document], list[dict]]:
        """解析 raw 全部 PDF → 切块 → 语料 jsonl 缓存。

        语法：fingerprint = 全部 PDF 的「文件名+mtime」摘要，写入缓存首行。
        原理：解析 660+ 页真实年报（含表格抽取）需数分钟，重启服务不该重付这笔钱；
             缓存命中时直接从 jsonl 还原 Document，跳过解析。
        返回：(docs, 每文档解析统计[{file,pages,blocks,table_blocks,chunks}])
        """
        cache = settings.parsed_dir / "corpus.jsonl"
        pdfs = sorted(settings.raw_dir.glob("*.pdf"))
        fp = hashlib.md5(
            ",".join(f"{p.name}:{p.stat().st_mtime_ns}" for p in pdfs).encode("utf-8")
        ).hexdigest()

        if cache.exists():
            lines = cache.read_text(encoding="utf-8").splitlines()
            if lines and json.loads(lines[0]).get("_fingerprint") == fp:
                docs = [
                    Document(page_content=r["text"], metadata=r["meta"])
                    for r in (json.loads(l) for l in lines[1:])
                ]
                return docs, []

        docs, stats = [], []
        for pdf in pdfs:
            parser = AnnualReportParser()
            blocks = parser.parse(pdf)
            import fitz
            pages = fitz.open(pdf).page_count
            meta = {}
            if "_" in pdf.stem:
                parts = pdf.stem.split("_")
                if len(parts) >= 2:
                    meta = {"company": parts[0],
                            "year": "".join(ch for ch in parts[1] if ch.isdigit())[:4]}
            chunks = FinancialChunker().split(blocks, doc_meta=meta)
            docs.extend(chunks)
            stats.append({"file": pdf.name, "pages": pages, "blocks": len(blocks),
                          "table_blocks": sum(1 for b in blocks if b.block_type == "table"),
                          "chunks": len(chunks)})
        # 写缓存：首行指纹，其后每行一个块
        with cache.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"_fingerprint": fp}, ensure_ascii=False) + "\n")
            for d in docs:
                f.write(json.dumps({"text": d.page_content, "meta": d.metadata},
                                   ensure_ascii=False) + "\n")
        return docs, stats

    def _parse_and_chunk(self, pdf_path: Path, doc_meta: dict[str, str] | None = None) -> list[Document]:
        """单个 PDF：解析 → 切块（/api/ingest 单文档路径用，不走缓存）。"""
        blocks = AnnualReportParser().parse(pdf_path)
        meta = dict(doc_meta or {})
        if not meta.get("company") and "_" in pdf_path.stem:
            parts = pdf_path.stem.split("_")
            if len(parts) >= 2:
                meta.setdefault("company", parts[0])
                meta.setdefault("year", "".join(ch for ch in parts[1] if ch.isdigit())[:4])
        return FinancialChunker().split(blocks, doc_meta=meta)

    def _build(self, docs: list[Document]) -> None:
        """由 docs 全量重建检索器与图。调用方负责持锁。"""
        self.retriever = HybridRetriever(docs)
        self.graph = build_rag_graph(self.retriever, checkpointer=self.checkpointer)

    # ---------- 入库：增量追加一个 PDF 并重建 ----------
    def ingest(self, pdf_path: Path, company: str = "", year: str = "") -> int:
        """解析新 PDF → 追加到语料 → 重建索引与图。返回该文档的分块数。"""
        meta = {k: v for k, v in {"company": company, "year": year}.items() if v}
        new_docs = self._parse_and_chunk(pdf_path, meta or None)
        with self.lock:
            old = self.retriever.docs if self.retriever else []
            # 幂等：同名文档先剔除旧块再追加，避免重复入库
            src = str(pdf_path)
            merged = [d for d in old if d.metadata.get("source") != src] + new_docs
            self._build(merged)
        return len(new_docs)

    # ---------- 高频问答缓存（Redis / fakeredis）----------
    _redis = None

    @classmethod
    def _cache_client(cls):
        """语法：redis_url 为空时自动降级 fakeredis（进程内实现，接口与 redis-py 完全一致）。
        用途：S6 完成标志——第二次问相同问题必须明显变快。"""
        if cls._redis is None:
            if settings.redis_url:
                import redis
                cls._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            else:
                import fakeredis
                cls._redis = fakeredis.FakeRedis(decode_responses=True)
        return cls._redis

    @staticmethod
    def cache_key(question: str) -> str:
        norm = "".join(question.split()).lower()
        return "rag:ans:" + hashlib.md5(norm.encode("utf-8")).hexdigest()

    def cache_get(self, question: str) -> dict | None:
        raw = self._cache_client().get(self.cache_key(question))
        return json.loads(raw) if raw else None

    def cache_set(self, question: str, payload: dict, ttl: int = 3600) -> None:
        self._cache_client().setex(self.cache_key(question), ttl, json.dumps(payload, ensure_ascii=False))


# 模块级单例：`from src.api.engine import engine`
engine = Engine()
