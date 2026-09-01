"""
BM25 稀疏检索（关键词通路）。

用途：混合召回的一路。与向量检索互补，财报场景下是刚需而非锦上添花。
原理：
  - 向量检索解决「同义表述」（净利润 vs 归属于母公司股东的净利润），
    但对「600519」这类证券代码、「2025年第三季度」这类精确时间短语不敏感。
  - BM25 靠字面精确匹配补上这个缺口，属于经典概率检索模型（Okapi BM25）。
  - 中文必须先分词：不分词的话整句会被当成一个巨型词项，效果崩坏。
    jieba 默认词典对财报术语切得稀碎（如「归属于母公司所有者的净利润」会被切碎），
    因此必须加载自定义财报词典。
"""
from __future__ import annotations

import re
from pathlib import Path

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.core.config import settings

# 财报核心术语（保证被 jieba 当成一个整体词，不被切碎）
_DEFAULT_TERMS = [
    "营业收入", "营业成本", "营业利润", "利润总额", "净利润", "归母净利润",
    "归属于母公司所有者的净利润", "归属于母公司股东的净利润", "扣非净利润",
    "扣除非经常性损益", "基本每股收益", "稀释每股收益", "加权平均净资产收益率",
    "净资产收益率", "总资产收益率", "毛利率", "净利率", "营业利润率",
    "资产负债率", "流动比率", "速动比率", "现金流量表", "资产负债表",
    "利润表", "经营活动产生的现金流量净额", "投资活动产生的现金流量净额",
    "筹资活动产生的现金流量净额", "货币资金", "应收账款", "存货",
    "商誉", "无形资产", "固定资产", "在建工程", "短期借款", "长期借款",
    "研发投入", "研发费用", "销售费用", "管理费用", "财务费用",
    "税金及附加", "资产减值损失", "信用减值损失", "公允价值变动损益",
    "投资收益", "其他收益", "营业外收入", "营业外支出", "所得税费用",
    "总资产", "总负债", "所有者权益", "股东权益", "每股净资产",
    "经营活动现金流", "自由现金流", "EBITDA", "ROE", "ROA",
    "应收账款周转率", "存货周转率", "总资产周转率",
    "合并报表", "母公司报表", "附注", "审计报告", "会计师事务所",
]

# 无意义字符：数字/符号噪声会让 BM25 的词项分布失真
_NOISE_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9%.\-万亿元]+")


def load_finance_dict(path: str | Path | None = None) -> None:
    """加载财报自定义词典到 jieba 全局词库。

    用途：让专业术语不被切碎，直接决定 BM25 与分词质量。
    原理：jieba.add_word 会把词条加入前缀词典，优先于默认切分规则。
    """
    terms = list(_DEFAULT_TERMS)
    p = Path(path) if path else settings.dict_path
    if p.exists():
        terms += [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    for t in terms:
        if t:
            jieba.add_word(t, freq=10000)  # 高词频确保强制成词


def tokenize(text: str) -> list[str]:
    """中文分词 + 噪声过滤。

    用途：BM25 索引与查询必须用同一套分词逻辑，否则词项对不上。
    原理：先用 jieba 精确模式切，再过滤纯符号噪声，保留数字（财报里数字是关键词）。
    """
    text = _NOISE_RE.sub(" ", text)
    return [w for w in jieba.lcut(text) if w.strip() and len(w) > 0]


class BM25Retriever:
    """BM25 检索器。用途：混合召回的稀疏通路。"""

    def __init__(self, k1: float | None = None, b: float | None = None):
        self.k1 = k1 if k1 is not None else settings.bm25_k1
        self.b = b if b is not None else settings.bm25_b
        self._bm25: BM25Okapi | None = None
        self._docs: list[Document] = []
        self._tokenized: list[list[str]] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def build(self, docs: list[Document]) -> "BM25Retriever":
        """构建索引。用途：入库时调用一次。

        原理：预先分词并缓存，查询时只需对 query 分词，避免全量重切。
        """
        load_finance_dict()
        self._docs = list(docs)
        self._tokenized = [tokenize(d.page_content) for d in docs]
        self._bm25 = BM25Okapi(self._tokenized, k1=self.k1, b=self.b)
        return self

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Document, float]]:
        """检索。返回 [(Document, score)]，按分数降序。

        用途：给混合召回提供稀疏通路结果。
        原理：BM25Okapi.get_scores 返回所有文档的相关性分，取 Top-K。
        """
        if not self.is_built:
            raise RuntimeError("索引未构建，请先调用 build()")
        k = top_k or settings.top_k_retrieve
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)  # type: ignore[union-attr]
        # argsort 全部再切片：文档量级（数千）下开销可忽略，代码更简单
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._docs[i], float(scores[i])) for i in ranked if scores[i] > 0]
