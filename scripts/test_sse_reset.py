"""
测试 A：SSE 协议测试——不跑真图，喂假事件序列，验证 _chat_gen 的翻译逻辑。

断言：
  1. check_hallucination 失败(passed=False)时，除了 status 事件还发 reset
  2. 全失败时 done.answer = fallback 话术（不是第三次 generate 的假 token 拼接）
  3. 缓存不写入（hallucination_passed=False）
  4. check 通过时不发 reset
"""
import asyncio
import json
import pathlib
import sys
from unittest.mock import MagicMock

# 允许直接 `python scripts/test_sse_reset.py` 运行
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.api.app import _chat_gen
from src.api.schemas import ChatRequest


def _fake_msg(content: str, node: str):
    """造一个 messages 事件的 payload：(AIMessageChunk-like, metadata)"""
    msg = MagicMock()
    msg.content = content
    meta = {"langgraph_node": node}
    return (msg, meta)


def _fake_updates(node: str, **kwargs):
    """造一个 updates 事件的 payload"""
    return {node: kwargs}


def make_fake_stream(scenario: str):
    """
    造一个 async generator，按 scenario 吐假事件序列。
    scenario = "all_fail"  或 "pass"
    """
    async def gen(*args, **kwargs):
        # route_query
        yield ("updates", _fake_updates("route_query", route="single",
                                         query_used="2025年营业收入", trace=["路由=single"]))
        # retrieve
        yield ("updates", _fake_updates("retrieve", documents=[], retrieve_rounds=1,
                                         trace=["检索第1轮"]))
        # grade_documents
        yield ("updates", _fake_updates("grade_documents", documents=[], trace=["过滤"]))

        if scenario == "all_fail":
            # ---- 三次 generate + 两次 check 失败 ----
            # 第 1 次 generate：吐假 token
            for chunk in ["营收", "9,999,999", ".99", "元"]:
                yield ("messages", _fake_msg(chunk, "generate"))
            # check 1 失败
            yield ("updates", _fake_updates("check_hallucination",
                                             hallucination_passed=False,
                                             trace=["数值校验失败, 原文缺失: ['9999999999.99']"]))
            # 第 2 次 generate：又吐假 token
            for chunk in ["又错", "123,456", "元"]:
                yield ("messages", _fake_msg(chunk, "generate"))
            # check 2 失败
            yield ("updates", _fake_updates("check_hallucination",
                                             hallucination_passed=False,
                                             trace=["数值校验失败, 原文缺失: ['123456']"]))
            # 第 3 次 generate：又吐假 token
            for chunk in ["还错", "789"]:
                yield ("messages", _fake_msg(chunk, "generate"))
            # check 3 失败 → fallback
            yield ("updates", _fake_updates("check_hallucination",
                                             hallucination_passed=False,
                                             trace=["数值校验失败"]))
            # fallback_answer 节点（纯函数，不产生 messages）
            yield ("updates", _fake_updates("fallback_answer",
                                             final_answer="⚠️ 检测到回答中的数值无法在原文中核实，为避免误导已拦截本次回答。",
                                             trace=["降级收尾"]))
        else:
            # ---- 一次通过 ----
            for chunk in ["营收", "1,098,765,432.10", "元"]:
                yield ("messages", _fake_msg(chunk, "generate"))
            yield ("updates", _fake_updates("check_hallucination",
                                             hallucination_passed=True,
                                             trace=["数值校验通过"]))

    return gen


def make_fake_final_state(scenario: str):
    """模拟 graph.get_state(config) 的返回值"""
    fs = MagicMock()
    if scenario == "all_fail":
        fs.values = {
            "final_answer": "⚠️ 检测到回答中的数值无法在原文中核实，为避免误导已拦截本次回答。建议：换个更具体的问题。",
            "answer": "还错789",  # 第三次 generate 的残留
            "hallucination_passed": False,
            "documents": [],  # fallback 不改 documents
        }
    else:
        fs.values = {
            "final_answer": "",
            "answer": "营收1,098,765,432.10元",
            "hallucination_passed": True,
            "documents": [],
        }
    return fs


async def run_scenario(scenario: str):
    from src.api import engine as engine_mod

    # 保存原始 graph
    orig_graph = engine_mod.engine.graph

    # 造假 graph
    fake_graph = MagicMock()
    fake_graph.astream = make_fake_stream(scenario)
    fake_graph.get_state = lambda config: make_fake_final_state(scenario)

    engine_mod.engine.graph = fake_graph
    engine_mod.engine.ready = True

    # 清掉缓存
    engine_mod.engine._cache_client().flushall()

    # mock store：fresh_session=True
    from src.api import store as store_mod
    orig_load = store_mod.load_history
    store_mod.load_history = lambda sid: []  # 空历史 = fresh session
    orig_append = store_mod.append_turn
    appended = []
    store_mod.append_turn = lambda *a, **kw: appended.append(a)

    try:
        req = ChatRequest(question="2025年营业收入是多少", session_id=f"test-{scenario}")
        events = []
        async for sse in _chat_gen(req):
            ev = sse["event"]
            data = json.loads(sse["data"])
            events.append((ev, data))

        return events, appended
    finally:
        engine_mod.engine.graph = orig_graph
        store_mod.load_history = orig_load
        store_mod.append_turn = orig_append


async def test_all_fail():
    print("=== 测试 A1：三次全失败 ===")
    events, appended = await run_scenario("all_fail")

    # 统计
    resets = [d for ev, d in events if ev == "reset"]
    done_events = [d for ev, d in events if ev == "done"]
    status_checks = [(ev, d) for ev, d in events if ev == "status" and d.get("node") == "check"]

    print(f"  reset 事件数: {len(resets)} (期望 3)")
    print(f"  check status 数: {len(status_checks)}")
    for ev, d in status_checks:
        print(f"    passed={d.get('passed')}")
    print(f"  done.answer = {done_events[0]['answer'][:80]}...")

    assert len(resets) == 3, f"期望 3 次 reset，实际 {len(resets)}"
    assert len(status_checks) == 3, f"期望 3 次 check status，实际 {len(status_checks)}"
    assert all(not d.get("passed") for ev, d in status_checks), "3 次 check 都应失败"
    assert "数值无法在原文中核实" in done_events[0]["answer"], \
        f"done.answer 不是 fallback 话术: {done_events[0]['answer'][:100]}"

    # 缓存不该写入
    from src.api import engine as engine_mod
    cached = engine_mod.engine.cache_get("2025年营业收入是多少")
    print(f"  cache_after = {cached}")
    assert cached is None, f"缓存不该有写入，但有: {cached}"

    print("  [OK] 通过\n")


async def test_pass():
    print("=== 测试 A2：一次通过 ===")
    events, appended = await run_scenario("pass")

    resets = [d for ev, d in events if ev == "reset"]
    done_events = [d for ev, d in events if ev == "done"]

    print(f"  reset 事件数: {len(resets)} (期望 0)")
    print(f"  done.answer = {done_events[0]['answer'][:80]}...")

    assert len(resets) == 0, f"一次通过不该有 reset，实际 {len(resets)}"
    assert "1,098,765,432.10" in done_events[0]["answer"]

    print("  [OK] 通过\n")


if __name__ == "__main__":
    asyncio.run(test_all_fail())
    asyncio.run(test_pass())
    print("=== 全部通过 ===")
