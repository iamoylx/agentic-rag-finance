"""
RAG 状态图组装（LangGraph 手写，禁用 create_react_agent —— 简历硬约束）。

用途：P1-S4 核心。把 Adaptive / Corrective / Self-RAG 三范式融进一张图。
图结构：
                      ┌─ direct ──► direct_answer ──► END
  route_query ────────┤
                      └─ single/multi ──► retrieve ──► grade_documents
                                                │
                       ┌────────────────────────┼──────────────────────┐
                       │有相关文档               │无相关且可重试         │无相关且重试上限
                       ▼                        ▼                      ▼
                   generate ◄──(web_documents) rewrite_query        web_search ─┐
                       │                        └────► retrieve ◄────┘           │
                       ▼                                                    │
                check_hallucination ◄───────────────────────────────────────┘
                       │
            ┌──────────┴───────────┐
            │通过                   │未通过且可重试
            ▼                      ▼
           END                generate（重新生成）
                                   │ 重试上限 → fallback_answer → END

防死循环：rewrite_count / retrieve_rounds / regen_count 三个计数器硬上限。
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.core.config import settings
from src.graph import nodes
from src.graph.state import RAGState
from src.retrieval.hybrid import HybridRetriever


def build_rag_graph(retriever: HybridRetriever, checkpointer=None):
    """构建并编译 RAG 图。

    参数：retriever 已建好索引的混合检索器；checkpointer 传 None 则无多轮持久化。
    """
    g = StateGraph(RAGState)

    # ---------- 注册节点 ----------
    g.add_node("route_query", nodes.route_query)
    g.add_node("direct_answer", nodes.direct_answer)
    g.add_node("retrieve", nodes.make_retrieve_node(retriever))
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("web_search", nodes.web_search)
    g.add_node("generate", nodes.generate)
    g.add_node("check_hallucination", nodes.check_hallucination)
    g.add_node("fallback_answer", nodes.fallback_answer)

    # ---------- 入口与直答分支（Adaptive RAG）----------
    g.set_entry_point("route_query")
    g.add_conditional_edges(
        "route_query",
        lambda s: "direct_answer" if s["route"] == "direct" else "retrieve",
        {"direct_answer": "direct_answer", "retrieve": "retrieve"},
    )
    g.add_edge("direct_answer", END)

    # ---------- Corrective RAG：检索后质量评估分流 ----------
    g.add_edge("retrieve", "grade_documents")  # 主链路：检索完必须评估相关性

    def after_grade(s: RAGState) -> str:
        if s.get("documents"):
            return "generate"
        if s.get("rewrite_count", 0) < settings.max_rewrite:
            return "rewrite_query"
        if not s.get("web_used"):
            return "web_search"
        return "fallback_answer"

    g.add_conditional_edges(
        "grade_documents", after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query",
         "web_search": "web_search", "fallback_answer": "fallback_answer"},
    )

    # 改写后再检索；联网结果直接送生成
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("web_search", "generate")

    # ---------- Self-RAG：生成后数值校验分流 ----------
    g.add_edge("generate", "check_hallucination")  # 主链路：生成完必须校验

    def after_check(s: RAGState) -> str:
        if s.get("hallucination_passed"):
            return "finalize"
        if s.get("regen_count", 0) < settings.max_regenerate:
            return "generate"          # 带着失败信息重新生成
        return "fallback_answer"

    # finalize 只做透传（把 answer 定格为 final_answer），保持图逻辑清晰
    g.add_node("finalize", lambda s: {"final_answer": s.get("answer", "")})
    g.add_conditional_edges(
        "check_hallucination", after_check,
        {"finalize": "finalize", "generate": "generate", "fallback_answer": "fallback_answer"},
    )
    g.add_edge("finalize", END)
    g.add_edge("fallback_answer", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


def answer(graph, question: str, thread_id: str = "default", callbacks=None) -> dict:
    """便捷调用：跑完整图并返回最终 state。

    thread_id 用于 checkpointer 区分会话（多轮追问「那2024年呢」依赖它）。
    callbacks 用于挂成本计量等观察者（如 src/core/cost.py 的 CostTracker）——
              走 LangChain 回调体系，业务节点无需感知。
    """
    config: dict = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    final = graph.invoke(
        {"question": question, "query_used": question,
         "retrieve_rounds": 0, "rewrite_count": 0, "regen_count": 0,
         "web_used": False, "trace": []},
        config=config,
    )
    return final
