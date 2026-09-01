"""
分块层：把解析后的 Block 切成适合检索的语义块。

用途：P1-S2 的前半段。分块质量直接决定 Recall@5 的上限。
原理（财报场景的三条硬规则）：
  1. 递归分块打底：按 [段落→句子→标点→字符] 的优先级递归切，尽量不破坏自然语义边界。
     中文 1 字 ≈ 1 token，512 字约等于英文 300 token，是召回精度与上下文完整性的较好平衡。
  2. 表格块禁止切碎：整表作为一个 chunk。表格被切成两半 = 表头丢失 = 答案必然错误。
     超大表格按行切分，但每片都重复携带表头。
  3. 块内注入上下文前缀：把「来源文件 + 页码 + 章节」拼到正文前。
     否则「营业收入」这个块在 3 家公司年报里向量几乎相同，检索无法区分主体。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings
from src.ingestion.pdf_parser import Block

# 中文友好分隔符优先级：先按段落，再按句号/分号，最后按字符兜底
_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]


@dataclass
class ChunkStats:
    """分块统计。用途：写进 METRICS.md，也是排查分块异常的第一手依据。"""

    n_blocks: int = 0
    n_chunks: int = 0
    n_table_chunks: int = 0
    avg_len: float = 0.0
    max_len: int = 0


class FinancialChunker:
    """财报专用分块器。"""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        inject_context: bool = True,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.inject_context = inject_context
        # length_function=len 对中文是按「字」计数，比按 token 更直观且无需额外依赖
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, blocks: list[Block], doc_meta: dict[str, str] | None = None) -> list[Document]:
        """把 Block 列表切成 Document 列表。

        用途：Ingestion 第二步，输出直接送向量化。
        参数：doc_meta 传入 {公司名, 年份} 等文档级信息，会注入到每个块的前缀。
        """
        doc_meta = doc_meta or {}
        docs: list[Document] = []

        for b in blocks:
            if b.block_type == "table":
                docs.extend(self._split_table(b, doc_meta))
            else:
                docs.extend(self._split_text(b, doc_meta))

        return docs

    def _context_prefix(self, b: Block, doc_meta: dict[str, str]) -> str:
        """构造上下文前缀。用途：让语义相近但主体不同的块能被区分开。"""
        parts = []
        if doc_meta.get("company"):
            parts.append(str(doc_meta["company"]))
        if doc_meta.get("year"):
            parts.append(f"{doc_meta['year']}年")
        if b.section:
            parts.append(b.section)
        parts.append(f"第{b.page}页")
        return "【" + " | ".join(parts) + "】\n"

    def _split_text(self, b: Block, doc_meta: dict[str, str]) -> list[Document]:
        """文本块：递归分块。"""
        prefix = self._context_prefix(b, doc_meta) if self.inject_context else ""
        texts = self._splitter.split_text(b.text)
        return [
            Document(
                page_content=prefix + t,
                metadata={
                    "page": b.page,
                    "source": b.source,
                    "block_type": "text",
                    "section": b.section,
                    **doc_meta,
                },
            )
            for t in texts
            if t.strip()
        ]

    def _split_table(self, b: Block, doc_meta: dict[str, str]) -> list[Document]:
        """表格块：优先整表单块；超长时按行切分但每片重复携带表头。

        原理：表头是表格的语义锚点。丢掉表头的表格片段对 LLM 是噪声，
              因此宁可重复存储表头，也不能让任何一片缺失它。
        """
        prefix = self._context_prefix(b, doc_meta) if self.inject_context else ""
        lines = [ln for ln in b.text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return [Document(page_content=prefix + b.text,
                             metadata={"page": b.page, "source": b.source,
                                       "block_type": "table", "section": b.section, **doc_meta})]

        header_lines = lines[:2]  # Markdown 表格的表头行 + 分隔行
        body_lines = lines[2:]
        full = "\n".join(lines)

        # 整表能放下就直接返回，保持表格完整
        if len(prefix) + len(full) <= self.chunk_size:
            return [Document(page_content=prefix + full,
                             metadata={"page": b.page, "source": b.source,
                                       "block_type": "table", "section": b.section, **doc_meta})]

        # 超长表格：按行分批，每批都带上表头
        docs: list[Document] = []
        budget = max(self.chunk_size - len(prefix) - len("\n".join(header_lines)) - 10, 200)
        batch: list[str] = []
        cur = 0

        def flush():
            nonlocal batch, cur
            if not batch:
                return
            content = prefix + "\n".join(header_lines + batch)
            docs.append(Document(
                page_content=content,
                metadata={"page": b.page, "source": b.source,
                          "block_type": "table", "section": b.section,
                          "table_partial": True, **doc_meta},
            ))
            batch, cur = [], 0

        for ln in body_lines:
            ln_len = len(ln) + 1
            if cur + ln_len > budget and batch:
                flush()
            batch.append(ln)
            cur += ln_len
        flush()
        return docs


def summarize_chunks(docs: list[Document]) -> ChunkStats:
    """统计分块结果。用途：入库前自检，异常（如表格块为 0）能立刻发现。"""
    lens = [len(d.page_content) for d in docs]
    return ChunkStats(
        n_chunks=len(docs),
        n_table_chunks=sum(1 for d in docs if d.metadata.get("block_type") == "table"),
        avg_len=round(sum(lens) / len(lens), 1) if lens else 0.0,
        max_len=max(lens) if lens else 0,
    )


def extract_section_hint(text: str) -> str:
    """从正文里猜章节名（如「第三节 财务报表」）。

    用途：pdfplumber/PyMuPDF 不直接给章节结构，用正则从文本里捞，
          用于上下文前缀注入，提升检索区分度。
    """
    m = re.search(r"^第[一二三四五六七八九十]+[节章篇]\s*[^\n]{0,30}", text, re.M)
    return m.group(0).strip() if m else ""
