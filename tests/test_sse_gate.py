"""测试 A：SSE 协议测试（用「假 stream」驱动完整 HTTP 链路）。

这个文件同时是对 dsh 那个问题的回答落地——
「_chat_gen 内部调用 engine.graph.astream(...)，你要替换的到底是哪个对象、
  替换成什么形状的东西？」

答案：
  · **_chat_gen 不需要被『喂』事件**。它是个 async generator，是**拉取式**的：
    测试方用 `async for frame in _chat_gen(req)` / 或用 HTTP 打它，由它主动去拉。
    所以要替换的不是「事件的注入点」，而是**它拉取的那个对象**。
  · 它拉取两个东西，都要假替身（这是最容易漏的一点）：
      ① `engine.graph.astream(state, config=, stream_mode=)` —— **异步迭代器**，
         逐个产出 `(mode, payload)` 二元组：
           mode == "updates"  → payload = {节点名: 该节点返回的状态增量 dict}
           mode == "messages" → payload = (消息块, metadata)，metadata["langgraph_node"]
                                标明来源节点，消息块 .content 是本次到达的 token 文本
      ② `engine.graph.get_state(config)` —— **同步**方法（代码里用 asyncio.to_thread 包了），
         返回对象要有 .values dict。它决定 done 事件里的最终答案。
    另外 `engine.cache_get` / `store.load_history` / `store.append_turn` / `engine.cache_set`
    也要给替身，否则会去碰真实缓存与数据库。

  · **替换哪个名字**：app.py 里写的是 `from src.api.engine import engine`，
    所以这个模块里绑定的是一个**名字** `src.api.app.engine`。
    必须 patch `src.api.app.engine`；patch `src.api.engine.engine` 是**无效的**
    （那只改了另一个模块的属性，app 里早已绑定的名字仍指向旧对象）。
    这是 Python `from x import y` 的经典坑，也是本测试最关键的一行。

为什么走 HTTP 而不直接调 _chat_gen：
  只有经过 EventSourceResponse 才验证到真正的 SSE 分帧（event:/data: 行、空行分隔）。
  顺带记住一个坑：sse-starlette 用 **\r\n\r\n** 分帧，按 "\n\n" 切会一帧都切不出来。
"""
from __future__ import annotations

import asyncio
import json
import re

import httpx

from src.api import app as app_module

# =====================================================================
# 假替身：形状严格对齐 langgraph astream(stream_mode=["updates","messages"])
# =====================================================================


class _Chunk:
    """消息块替身：只需要 .content（app.py 里就是取 content 当 token 文本）。"""

    def __init__(self, content: str) -> None:
        self.content = content


class _Doc:
    """Document 替身：只需要 .page_content 与 .metadata（build_corpus_norm / _citations 用）。"""

    def __init__(self, text: str, page: int = 5) -> None:
        self.page_content = text
        self.metadata = {"page": page, "source": "贵州茅台_2024.pdf"}


class _StateView:
    """get_state 的返回值替身：只需要 .values。"""

    def __init__(self, values: dict) -> None:
        self.values = values


class _FakeGraph:
    def __init__(self, events: list, final_values: dict) -> None:
        self._events = events
        self._final = final_values

    async def astream(self, init_state, config=None, stream_mode=None):
        """必须是 async generator：调用即得异步迭代器，逐个吐 (mode, payload)。"""
        for mode, payload in self._events:
            yield mode, payload

    def get_state(self, config):          # 同步：调用方用 asyncio.to_thread 包它
        return _StateView(self._final)


class _FakeEngine:
    def __init__(self, events: list, final_values: dict) -> None:
        self.graph = _FakeGraph(events, final_values)
        self.ready = True

    def cache_get(self, question):        # 走慢路（不吃缓存）
        return None

    def cache_set(self, question, payload, ttl=3600):
        return None


# =====================================================================
# 事件脚本构造
# =====================================================================


def _tok(text: str, node: str = "generate"):
    return ("messages", (_Chunk(text), {"langgraph_node": node}))


def _upd(node: str, delta: dict):
    return ("updates", {node: delta})


def _script(tokens: list[str], docs: list, *, node: str = "generate",
            passed: bool = True, final: dict | None = None):
    """拼一条完整脚本：路由 → 检索 → 打分 → 逐 token 生成 → 数值校验。"""
    events = [
        _upd("route_query", {"route": "single", "query_used": "q"}),
        _upd("retrieve", {"documents": docs, "retrieve_rounds": 1}),
        _upd("grade_documents", {"documents": docs}),
    ]
    events += [_tok(t, node) for t in tokens]
    events += [_upd("check_hallucination", {"hallucination_passed": passed})]
    values = final if final is not None else {"final_answer": "".join(tokens), "documents": docs}
    return events, values


# =====================================================================
# SSE 解析 + HTTP 驱动
# =====================================================================


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """按 SSE 规范切帧。

    注意：sse-starlette 用 \\r\\n\\r\\n 分帧，只按 "\\n\\n" 切会得到 0 帧。
    """
    frames = []
    for block in re.split(r"\r?\n\r?\n", text):
        ev = data = None
        for line in block.splitlines():
            m = re.match(r"^(event|data):\s?(.*)$", line)
            if not m:
                continue
            if m.group(1) == "event":
                ev = m.group(2)
            else:
                data = m.group(2)
        if ev and data is not None:
            frames.append((ev, json.loads(data)))
    return frames


def _post(question: str) -> list[tuple[str, dict]]:
    """用 ASGITransport 直接打 ASGI 应用（不起真实端口），读回整段 SSE。"""

    async def _run():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/chat",
                                     json={"question": question, "session_id": "t-sse"})
            assert resp.status_code == 200, resp.text
            return _parse_sse(resp.text)

    return asyncio.run(_run())


def _install(monkeypatch, events, final_values):
    """把 app 模块里的 engine / store 换成替身。

    ★ 关键：patch 的是 `src.api.app.engine`（模块内已绑定的那个名字），
      不是 `src.api.engine.engine`——后者改的是另一个模块的属性，对本模块无效。
    """
    monkeypatch.setattr(app_module, "engine", _FakeEngine(events, final_values))
    monkeypatch.setattr(app_module.store, "load_history", lambda tid: [])
    monkeypatch.setattr(app_module.store, "append_turn", lambda *a, **k: None)


def _streamed(frames) -> str:
    """把 token 帧拼起来 = 用户实际看到的正文。"""
    return "".join(d["t"] for e, d in frames if e == "token")


# =====================================================================
# 用例
# =====================================================================

DOCS = [_Doc("贵州茅台2024年营业收入为170,899,152,276.34元。", page=5)]


def test_happy_path_number_streams_through(monkeypatch):
    """正常路径：数字在语料里 → 照常上屏，且仍是逐 token 流式（没被整段扣留）。"""
    tokens = ["贵州茅台", "2024年", "营业收入为", "170,", "899,152,", "276.34", "元。"]
    events, values = _script(tokens, DOCS)
    _install(monkeypatch, events, values)

    frames = _post("贵州茅台2024年营业收入是多少？")
    shown = _streamed(frames)

    assert shown == "".join(tokens)          # 内容一字不少
    tok_frames = [f for f in frames if f[0] == "token"]
    # 正文是分多帧到达的：证明闸门只扣「正在拼的数字」，不是整段缓冲
    assert len(tok_frames) >= 4, f"token 帧只有 {len(tok_frames)} 个，疑似被整段扣留"
    assert not [f for f in frames if f[0] == "reset"]   # 通过 → 不该 reset
    assert [f for f in frames if f[0] == "done"][0][1]["answer"] == "".join(tokens)


def test_fabricated_number_never_reaches_screen(monkeypatch):
    """★ 核心用例：编造的数字（语料里没有 999,888,000）绝不出现在任何 token 帧里。

    同时验证「不整段扣留」：错数字之前的正文照常逐帧流出。
    """
    tokens = ["贵州茅台", "2024年", "营业收入为", "999,", "888,000", "元。"]
    events, values = _script(tokens, DOCS, passed=False,
                             final={"final_answer": "⚠️ 检测到回答中的数值无法在原文中核实。",
                                    "documents": DOCS})
    _install(monkeypatch, events, values)

    frames = _post("贵州茅台2024年营业收入是多少？")
    shown = _streamed(frames)

    # 1) 编造的数字一个字符都没上屏
    assert "999" not in shown and "888" not in shown, f"编造数字泄漏到屏幕: {shown!r}"
    # 2) 但正文照常流式（说明没有为了安全把整段答案扣到校验之后）
    assert "营业收入为" in shown
    assert shown.count("贵州茅台") == 1
    # 3) 校验失败 → 发了 reset（前端据此清空气泡）
    assert [f for f in frames if f[0] == "reset"], "未通过校验却没发 reset"
    # 4) done 里的答案是降级话术，不含编造数字
    done = [f for f in frames if f[0] == "done"][0][1]
    assert "999" not in done["answer"] and "888" not in done["answer"]


def test_direct_answer_is_not_gated(monkeypatch):
    """边界用例：direct 分支不挂 check_hallucination，因此**不能**过闸门。

    若给 direct_answer 也加闸门，它没有语料可比对 → 任何数字都被判「编造」→
    屏幕上莫名为空，而权威校验从未对它提出过这个要求（过度拦截）。
    """
    tokens = ["市盈率", "通常", "为", "20", "倍。"]
    # 语料为空：direct 分支不检索
    events = [_upd("route_query", {"route": "direct", "query_used": "q"})]
    events += [_tok(t, "direct_answer") for t in tokens]
    _install(monkeypatch, events, {"final_answer": "".join(tokens), "documents": []})

    frames = _post("什么是市盈率？")
    shown = _streamed(frames)

    assert shown == "".join(tokens), "direct_answer 被误拦了（不该过闸门）"
    assert "20" in shown


def test_trailing_number_is_flushed(monkeypatch):
    """收尾用例：答案以数字结尾时，扣住的尾巴必须由 flush 放行，否则屏幕少一截。"""
    tokens = ["基本每股收益为", "68.64"]      # 最后一个 token 就是数字，无后续非数字字符
    docs = [_Doc("基本每股收益为68.64元/股。", page=5)]
    events, values = _script(tokens, docs)
    _install(monkeypatch, events, values)

    frames = _post("基本每股收益是多少？")
    assert _streamed(frames) == "基本每股收益为68.64"


def test_gate_does_not_leak_between_regenerations(monkeypatch):
    """回归用例：第一次生成被闸门拦下后，重生成必须用**干净的新闸门**。

    若闸门状态（violated / 扣留缓冲）跨轮泄漏，第二轮会把本来合法的数字也吞掉。
    """
    good_docs = [_Doc("贵州茅台2024年营业收入为170,899,152,276.34元。", page=5)]
    events = [
        _upd("route_query", {"route": "single", "query_used": "q"}),
        _upd("retrieve", {"documents": good_docs, "retrieve_rounds": 1}),
        _upd("grade_documents", {"documents": good_docs}),
        # 第 1 轮：编造数字
        _tok("贵州茅台"), _tok("营业收入为"), _tok("999"), _tok("元。"),
        _upd("check_hallucination", {"hallucination_passed": False}),
        # 第 2 轮：改对了（数字在语料里）
        _tok("贵州茅台"), _tok("营业收入为"), _tok("170,"), _tok("899,152,276.34"), _tok("元。"),
        _upd("check_hallucination", {"hallucination_passed": True}),
    ]
    final = {"final_answer": "贵州茅台营业收入为170,899,152,276.34元。", "documents": good_docs}
    _install(monkeypatch, events, final)

    frames = _post("贵州茅台2024年营业收入是多少？")
    shown = _streamed(frames)

    assert "999" not in shown, "第一轮的编造数字泄漏"
    assert "170,899,152,276.34" in shown, "重生成后的合法数字被闸门误吞（状态泄漏）"
    assert len([f for f in frames if f[0] == "reset"]) == 1
