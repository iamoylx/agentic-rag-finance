"""
LangGraph 状态定义（RAG 全局状态）。

用途：所有节点读写的唯一状态容器。TypedDict 方式，字段即节点间的数据契约。
原理：
  - 不用 add_messages：RAG 流水线是「工作流」而非「对话流」，中间态（路由/文档/校验结果）
    不是消息，用显式字段更清晰，也避免历史消息膨胀。
  - 所有计数器字段（rewrite_count 等）是防死循环的关键，条件边依赖它们判停。
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    # ---------- 输入 ----------
    question: str
    history: list[dict[str, str]]    # 近几轮问答 [{"q":..,"a":..}]，多轮追问的指代消解

    # ---------- Adaptive RAG：路由决策 ----------
    # direct: 定义/常识类，不检索直答；single: 一次检索；multi: 需要拆解多跳
    route: Literal["direct", "single", "multi"]

    # ---------- 检索中间态 ----------
    documents: list[Document]        # 当前轮检索并过滤后的文档
    query_used: str                  # 实际用于检索的 query（可能是改写后的）
    retrieve_rounds: int             # 已检索轮次
    rewrite_count: int               # 已改写次数
    web_used: bool                   # 是否动用了联网搜索

    # ---------- 生成与校验 ----------
    answer: str
    citations: list[dict]            # [{"page": int, "source": str, "quote": str}]
    hallucination_passed: bool
    regen_count: int                 # 幻觉校验不通过后的重生成次数

    # ---------- 收尾 ----------
    final_answer: str                # 经降级处理后的最终回复（含"未找到"话术）
    # Annotated + operator.add：多节点并行/顺序执行时 trace 追加而非覆盖
    trace: Annotated[list[str], operator.add]  # 执行轨迹（前端展示 / 调试用）
    # 写入方式统一为「节点在返回的增量 dict 里带 trace: [msg]」——
    # 不要在节点内原地 mutate（LangGraph 以返回值为准做 channel 合并，原地改不可靠）
