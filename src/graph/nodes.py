"""
RAG 图的全部节点实现（每节点 = 一个纯函数：state 增量 -> state 增量）。

对应简历三个自纠正范式：
  Adaptive RAG   -> route_query（查询分类路由）
  Corrective RAG -> grade_documents / rewrite_query / web_search（检索质量评估与降级）
  Self-RAG       -> generate + check_hallucination（生成后数值一致性校验）

设计约定：
  - 节点只做「读 state -> 计算 -> 返回增量 dict」，不持有跨调用状态；
    模型等重资源由 rag_graph.py 构建时以闭包/参数注入。
  - LLM 输出一律 JSON：解析失败时给出保守默认值，绝不让一次解析失败炸掉整张图。
"""
from __future__ import annotations

import json
import re

from langchain_core.documents import Document

from src.core.config import settings
from src.core.llm import get_cheap_llm, get_llm
from src.graph.state import RAGState, trace
from src.retrieval.hybrid import HybridRetriever

_NUM_RE = re.compile(r"\d[\d,，]*(?:\.\d+)?")


def _norm_num(s: str) -> str:
    """数字归一化：去千分位/中文逗号，便于「答案数字 vs 原文数字」匹配。"""
    return s.replace(",", "").replace("，", "").rstrip(".")


def _parse_json(text: str) -> dict | list | None:
    """宽容 JSON 解析：剥掉 markdown 代码围栏后尝试解析。

    用途：Agnes/DeepSeek 偶尔会包 ```json ...```，直接 json.loads 会失败。
    """
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        return None


# =====================================================================
# 1) Adaptive RAG：路由
# =====================================================================

ROUTE_PROMPT = """你是检索路由器。根据问题类型选择处理方式：
- "direct": 通用定义/常识/概念解释类问题（如"什么是商誉"），无需检索文档即可回答
- "single": 能用一次检索回答的事实性问题（如"公司2025年营收是多少"）
- "multi": 需要对比或多步推导的复杂问题（如"对比两家公司的毛利率"）

若给定了对话历史且当前问题含指代（如"那2024年呢"），必须先把问题改写成
不依赖上下文的独立问题填入 standalone（如"公司2024年营收是多少"）；
无指代时 standalone 原样复制问题。

只输出 JSON：{{"route": "direct|single|multi", "standalone": "独立化后的问题", "reason": "一句话理由"}}
{history}问题：{q}"""


def _history_block(state: RAGState, turns: int = 3) -> str:
    """把近几轮问答拼成 prompt 片段。用途：多轮追问的指代消解（checkpointer 的 thread_id 串会话）。"""
    history = state.get("history") or []
    if not history:
        return ""
    lines = [f"对话历史（仅供理解指代）：" ]
    for h in history[-turns:]:
        lines.append(f"问：{h.get('q', '')}\n答：{h.get('a', '')[:200]}")
    return "\n".join(lines) + "\n"


def route_query(state: RAGState) -> dict:
    q = state["question"]
    llm = get_cheap_llm()
    resp = llm.invoke(ROUTE_PROMPT.format(q=q, history=_history_block(state))).content
    data = _parse_json(resp) or {}
    route = data.get("route")
    if route not in ("direct", "single", "multi"):
        route = "single"  # 保守默认：宁可多检索，不要漏检索
    # standalone：独立化后的问题；后续检索/改写/生成都以它为准
    standalone = (data.get("standalone") or "").strip() or q
    return {"route": route, "query_used": standalone,
            "trace": [f"路由={route}({data.get('reason', '')[:30]})" + (f"，独立化→{standalone}" if standalone != q else "")]}


# =====================================================================
# 2) 直答（不检索）
# =====================================================================

def direct_answer(state: RAGState) -> dict:
    resp = get_llm().invoke(
        f"简明回答问题（这是概念/常识类问题，无需引用财报）：\n{state['question']}"
    )
    return {"final_answer": resp.content.strip(), "citations": [],
            "trace": ["direct 直答，未检索"]}


# =====================================================================
# 3) 检索
# =====================================================================

def make_retrieve_node(retriever: HybridRetriever):
    def retrieve(state: RAGState) -> dict:
        q = state.get("query_used") or state["question"]
        # search 返回 [(Document, score)]，图状态只保留 Document 本体
        hits = retriever.search(q, top_k=settings.top_k_final)
        docs = [d for d, _ in hits]
        rounds = state.get("retrieve_rounds", 0) + 1
        return {"documents": docs, "query_used": q, "retrieve_rounds": rounds,
                "trace": [f"检索第{rounds}轮: '{q[:24]}' 命中{len(docs)}块"]}
    return retrieve


# =====================================================================
# 4) Corrective RAG：检索质量评估
# =====================================================================

GRADE_PROMPT = """判断以下每个文档片段与问题的相关性。只输出 JSON 数组，每项 {{"i": 序号, "relevant": true/false}}。

问题：{q}
文档片段：
{docs}"""


def grade_documents(state: RAGState) -> dict:
    docs = state.get("documents", [])
    if not docs:
        return {"documents": []}
    q = state.get("query_used") or state["question"]

    listing = "\n".join(
        f"[{i}] (第{d.metadata.get('page')}页) {d.page_content[:180]}"
        for i, d in enumerate(docs)
    )
    resp = get_cheap_llm().invoke(GRADE_PROMPT.format(q=q, docs=listing)).content
    data = _parse_json(resp)
    verdicts: dict[int, bool] = {}
    if isinstance(data, list):
        for item in data:
            try:
                verdicts[int(item["i"])] = bool(item["relevant"])
            except Exception:
                continue

    kept = [d for i, d in enumerate(docs) if verdicts.get(i, True)]  # 解析失败视为相关
    dropped = len(docs) - len(kept)
    return {"documents": kept,
            "trace": [f"相关性过滤: 保留{len(kept)}块, 剔除{dropped}块"]}


# =====================================================================
# 5) Corrective RAG：查询改写
# =====================================================================

REWRITE_PROMPT = """当前检索query召回质量差。请改写以提高召回：
- 补全可能缺失的主体/年份/限定词
- 把口语化表达改成财报术语（如"赚了多少钱"->"净利润"）
- 只输出改写后的query本身，不要任何解释。

原query：{q}
用户原始问题：{orig}"""


def rewrite_query(state: RAGState) -> dict:
    q = state.get("query_used") or state["question"]
    resp = get_cheap_llm().invoke(
        REWRITE_PROMPT.format(q=q, orig=state["question"])
    ).content
    new_q = resp.strip().strip('"').splitlines()[0][:120]
    return {"query_used": new_q, "rewrite_count": state.get("rewrite_count", 0) + 1,
            "trace": [f"改写query: '{q[:24]}' -> '{new_q[:30]}'"]}


# =====================================================================
# 6) Corrective RAG：联网降级（Tavily）
# =====================================================================

def web_search(state: RAGState) -> dict:
    from tavily import TavilyClient

    q = state.get("query_used") or state["question"]
    client = TavilyClient(api_key=settings.tavily_api_key)
    results = client.search(q, max_results=settings.web_search_max_results)
    docs = [
        Document(
            page_content=f"（联网搜索结果，仅供参考）{r.get('content', '')[:500]}",
            metadata={"source": r.get("url", "web"), "page": 0, "block_type": "web"},
        )
        for r in results.get("results", [])
    ]
    return {"documents": docs, "web_used": True,
            "trace": [f"联网降级: Tavily 返回{len(docs)}条"]}


# =====================================================================
# 7) 生成（带页码引用）
# =====================================================================

GEN_PROMPT = """你是财报分析助手。仅根据提供的原文片段回答问题。

规则：
1. 只能使用片段中的信息，禁止编造。
2. 片段不足以回答时，回复"原文中未找到相关信息"。
3. 【答非所问禁令】问的是指标 A、片段里只有指标 B 时，回复"原文中未找到相关信息"，
   禁止列举 B 指标（哪怕同页/同类）。
   例：问"光伏组件销量"，片段只有"动力电池销量 381GWh" → 必须回复
   "原文中未找到相关信息"，绝不能输出 381GWh。
4. 【禁止单位换算】必须逐字使用原文中的数字与单位（如原文是"170,899,152,276.34元"
   就原样引用，禁止换算成"1,708.99亿元"）。
5. 答案末尾标注引用页码，格式：[引用: 第X页]（多页用顿号分隔）；
   若依据的是网络来源片段，标注 [来源: 网络]。

问题：{q}

原文片段：
{docs}"""


def generate(state: RAGState) -> dict:
    docs = state.get("documents", [])
    q = state.get("query_used") or state["question"]
    if not docs:
        return {"answer": "原文中未找到相关信息", "citations": []}

    # Self-RAG 重生成：上轮数值校验失败时，把失败数字注入 prompt 修正采样
    regen = state.get("regen_count", 0)
    hint = ""
    if state.get("hallucination_passed") is False:
        regen += 1
        prev_nums = sorted({_norm_num(m.group()) for m in _NUM_RE.finditer(state.get("answer", ""))},
                           key=len, reverse=True)[:3]
        hint = (f"\n⚠️ 上次回答中的数字 {prev_nums} 未在原文出现，属于编造。"
                "本次必须逐字使用原文中的数字，无法核实的信息直接说「原文未找到」。")

    def _frag_header(i: int, d: Document) -> str:
        """片段头：年报块带页码，网络来源块不带假页码（否则 LLM 会编造"第0页"）。"""
        if d.metadata.get("block_type") == "web":
            return f"[片段{i} | 网络来源]"
        return f"[片段{i} | 第{d.metadata.get('page')}页]"

    context = "\n\n".join(
        _frag_header(i, d) + "\n" + d.page_content
        for i, d in enumerate(docs, 1)
    )
    history_block = _history_block(state)
    prompt = history_block + GEN_PROMPT.format(q=q, docs=context) + hint
    resp = get_llm().invoke(prompt).content.strip()
    # 网关容错：OpenAI 兼容网关偶发返回空 content（不报错），重试一次，
    # 仍为空则按「未找到」兜底，绝不让空串流进状态图（空串会被数值校验误判通过）
    if not resp:
        resp = get_llm().invoke(prompt).content.strip()
    if not resp:
        resp = "原文中未找到相关信息"

    # 从回答中解析引用：年报块解析页码，网络来源块解析 [来源: 网络]
    pages = sorted({int(p) for p in re.findall(r"第(\d+)页", resp)})
    citations = [
        {"page": p,
         "source": next((d.metadata.get("source") for d in docs
                         if d.metadata.get("page") == p), ""),
         "quote": next((d.page_content[:80] for d in docs
                        if d.metadata.get("page") == p), "")}
        for p in pages
    ]
    web_docs = [d for d in docs if d.metadata.get("block_type") == "web"]
    if web_docs and ("来源: 网络" in resp or "来源：网络" in resp):
        citations.append({"page": -1, "source": "web",
                          "url": web_docs[0].metadata.get("source", ""),
                          "quote": web_docs[0].page_content[:80]})
    return {"answer": resp, "citations": citations, "regen_count": regen,
            "trace": [f"生成完成{f'(第{regen}次重生成)' if regen else ''}"]}


# =====================================================================
# 8) Self-RAG：数值一致性校验（规则法，不走 LLM）
# =====================================================================

def check_hallucination(state: RAGState) -> dict:
    """答案中出现的数字必须能在引用的原文里找到（归一化后），否则判为幻觉。

    原理：金融场景数值错报是事故级风险，靠「生成后回溯校验」兜底而非指望模型自觉。
          纯规则实现：零 LLM 成本、可解释、可单测。
    """
    answer = state.get("answer", "")
    docs = state.get("documents", [])
    corpus = "\n".join(d.page_content for d in docs)
    corpus_norm = _norm_num(corpus)

    nums = {_norm_num(m.group()) for m in _NUM_RE.finditer(answer)}
    # 过滤噪声：页码引用（通常1-3位且伴随"第X页"）与序号不计入
    meaningful = {n for n in nums if len(n.replace(".", "")) >= 2}
    missing = [n for n in meaningful if n not in corpus_norm]

    passed = not missing
    msg = "数值校验通过" if passed else f"数值校验失败, 原文缺失: {missing[:3]}"
    return {"hallucination_passed": passed, "trace": [msg]}


# =====================================================================
# 9) 降级收尾（幻觉改不动 / 多轮检索全失败）
# =====================================================================

def fallback_answer(state: RAGState) -> dict:
    ans = state.get("answer", "")
    if state.get("hallucination_passed") is False:
        final = (
            "⚠️ 检测到回答中的数值无法在原文中核实，为避免误导已拦截本次回答。\n"
            "建议：换个更具体的问题（补充公司/年份/指标名），或直接查看对应年报原文。"
        )
    else:
        final = (
            "原文中未找到相关信息。可尝试：\n"
            "1. 换用财报标准术语（如\"归母净利润\"而非\"赚了多少钱\"）；\n"
            "2. 明确年份与主体；\n3. 使用联网结果（如已启用）。"
        )
    note = ans if ans and "未找到" not in ans else ""
    return {"final_answer": final, "trace": ["降级收尾" + (f": {note[:30]}" if note else "")]}
