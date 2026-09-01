"""
年报 PDF 解析层：把 PDF 转成「带页码、区分文本/表格」的结构化块。

用途：P1-S1。输出是后续分块、向量化、引用溯源的唯一数据源。
原理（三条关键设计）：
  1. 表格提取用 PyMuPDF 原生 find_tables().extract()：检测与内容提取出自同一引擎，
     字符坐标天然一致，避免了「pdfplumber 按字符坐标重建单元格导致数字错位」的经典问题。
     （实测：pdfplumber 会把 12,345,678,901.23 切成 12,345,678,90 + 11.02 这样的碎片）
  2. 表格转 Markdown：LLM 对 Markdown 表格的理解远好于空格对齐的纯文本，
     直接决定后续表格型问题的回答准确率。
  3. 页码是整个项目的命脉：后面「强制引用页码」和「数值一致性校验」全靠 page 字段，
     这里丢掉页码，后面全部功能做不出来。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


@dataclass
class Block:
    """一个解析后的内容块。

    用途：统一的中间表示，贯穿解析→分块→入库→引用全流程。
    page 从 1 开始（与 PDF 阅读器显示一致，方便人工核对与前端跳转）。
    """

    text: str
    page: int
    block_type: str = "text"          # text | table
    section: str = ""                  # 所属章节（用于块内上下文注入）
    source: str = ""
    bbox: tuple[float, float, float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "page": self.page,
            "block_type": self.block_type,
            "section": self.section,
            "source": self.source,
        }


# 页眉页脚常见模式：公司名+年报、纯页码、"第 X 页 共 Y 页" 等
_HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\s*[-—–]*\s*\d+\s*[-—–]*\s*$"),                 # 纯页码
    re.compile(r"^第?\s*\d+\s*页?\s*(/|共|of)?\s*\d*\s*页?\s*$"),  # 第X页/共Y页
    re.compile(r"^.*?(年度报告|半年度报告|招股说明书)\s*$"),          # XX公司年度报告
    re.compile(r"^.*?(财务报表附注|审计报告)\s*[-—–]*\s*\d+\s*$"),
]


def _is_header_footer(line: str) -> bool:
    """判断一行是否为页眉/页脚。用途：清洗重复噪声，避免污染向量库。"""
    s = line.strip()
    if not s or len(s) > 60:
        return False
    return any(p.match(s) for p in _HEADER_FOOTER_PATTERNS)


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """把 pdfplumber 提取的二维表格转成 Markdown 表格。

    用途：让表格内容对 LLM 友好。
    原理：首行作表头，其余作数据行；空单元格补空串，避免 Markdown 结构错位。
    """
    if not table or not table[0]:
        return ""
    rows = [[(c or "").strip().replace("|", "\\|") for c in row] for row in table]
    # 丢弃全空行
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return ""

    header = rows[0]
    # 表头若全为空（常见于无表头表格），用占位列名保证 Markdown 合法
    if not any(header):
        header = [f"列{i + 1}" for i in range(len(header))]
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows[1:]:
        # 补齐/截断到表头长度，防止行宽不一致破坏表格结构
        r = (r + [""] * len(header))[: len(header)]
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


class AnnualReportParser:
    """年报 PDF 解析器（混合策略）。"""

    def __init__(self, extract_tables: bool = True):
        # 用途：开关控制。表格提取很慢（pdfplumber 逐页解析），
        #       原型阶段可关掉快速验证文本链路。
        self.extract_tables = extract_tables

    def parse(self, pdf_path: str | Path) -> list[Block]:
        """解析单个 PDF，返回 Block 列表。

        用途：Ingestion 主入口。
        原理：逐页处理 → 先取表格区域 → 表格区域外的文本按块提取 → 清洗页眉页脚。
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

        blocks: list[Block] = []
        with fitz.open(pdf_path) as doc:
            n_pages = doc.page_count
            for idx in range(n_pages):
                page = doc[idx]
                page_no = idx + 1  # 用户可见页码从 1 开始

                # 1) 表格：检测 + 提取一次完成（同一引擎，坐标一致）
                table_zones: list[tuple[float, float, float, float]] = []
                if self.extract_tables:
                    try:
                        tables = page.find_tables()
                        for t in tables:
                            table_zones.append(tuple(t.bbox))  # type: ignore[arg-type]
                            md = _table_to_markdown(t.extract())
                            if md:
                                blocks.append(Block(
                                    text=md,
                                    page=page_no,
                                    block_type="table",
                                    source=pdf_path.name,
                                ))
                    except Exception:
                        table_zones = []

                # 2) 文本块：跳过落在表格区域内的部分，避免与表格重复入库
                raw = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
                for b in raw:
                    x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                    if not text or not text.strip():
                        continue
                    if self._in_any_zone((x0, y0, x1, y1), table_zones):
                        continue
                    # get_text("blocks") 可能把图片块也返回（block_type=1），过滤掉
                    if len(b) > 6 and b[6] == 1:
                        continue
                    cleaned = self._clean_text(text)
                    if cleaned:
                        blocks.append(Block(
                            text=cleaned,
                            page=page_no,
                            block_type="text",
                            source=pdf_path.name,
                            bbox=(x0, y0, x1, y1),
                        ))

        return blocks

    @staticmethod
    def _in_any_zone(
        bbox: tuple[float, float, float, float],
        zones: list[tuple[float, float, float, float]],
        iou_threshold: float = 0.3,
    ) -> bool:
        """判断文本块是否落在表格区域内（按面积重叠比例）。

        用途：避免表格内容既以表格形式入库、又以文本形式重复入库。
        原理：计算两个矩形的交集面积 / 文本块自身面积，超过阈值即视为表格内容。
        """
        if not zones:
            return False
        ax0, ay0, ax1, ay1 = bbox
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        if area_a <= 0:
            return False
        for z in zones:
            ix0, iy0, ix1, iy1 = max(ax0, z[0]), max(ay0, z[1]), min(ax1, z[2]), min(ay1, z[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if inter / area_a >= iou_threshold:
                return True
        return False

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗单块文本：去页眉页脚、合并断行、压缩空白。

        原理：PDF 抽取常出现「中文行内被断行」与「多余空行」，
              前者要合并（否则分块会切断语义），后者要压缩（否则浪费 token）。
        """
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln and not _is_header_footer(ln)]
        if not lines:
            return ""
        merged = "".join(lines) if _is_cjk_dense(text) else " ".join(lines)
        return re.sub(r"\s{2,}", " ", merged).strip()


def _is_cjk_dense(text: str) -> bool:
    """判断文本是否以中文为主。

    用途：中文 PDF 断行是排版产物，应直接拼接（不加空格）；
          英文断行则需补空格，否则单词会粘连。
    """
    sample = text[:200]
    cjk = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(len(sample), 1) > 0.3
