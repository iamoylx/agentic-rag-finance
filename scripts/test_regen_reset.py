"""
验证重生成 reset 事件 + 缓存不污染。
patch get_llm 让 generate 永远返回含假数字的答案 → check 永远失败 → 走满 max_regenerate=2 → fallback。

★ reset 次数为什么是 3 而不是 2（易错点，已实证）：
  max_regenerate=2 是「**重生成**次数上限」，不含首答。generate 里的计数器
  regen_count 只在「上一轮校验失败」时 +1（见 nodes.py::generate），所以实际生成次数是：
      gen#1  首答        regen_count=0 → check 失败 → 0<2 → 回到 generate   ← 屏幕上有假答案
      gen#2  第1次重生成  regen_count=1 → check 失败 → 1<2 → 回到 generate   ← 屏幕上有假答案
      gen#3  第2次重生成  regen_count=2 → check 失败 → 2<2 不成立 → fallback ← 屏幕上有假答案
  共 1 + max_regenerate = 3 次生成、3 次 check 失败 → **必须发 3 次 reset**。
  若只发 2 次，第 3 次的假答案无人擦除，fallback 话术会直接拼接在假数字后面 → 屏幕污染。
  所以次数 = settings.max_regenerate + 1，写死数字会在改配置时静默失配。

断言：
  1. SSE 流收到 max_regenerate+1 次 reset 事件
  2. done 事件的 answer 等于 fallback 话术（不是第三次 generate 的假数字）
  3. store 历史里存的是规范答案（fallback 话术）
  4. Redis 缓存里没有写入（hallucination_passed=False）
"""
import asyncio
import json
import pathlib
import sys
from unittest.mock import MagicMock

# 允许直接 `python scripts/test_regen_reset.py` 运行（否则 sys.path[0]=scripts/，import src 失败）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.api.engine import engine
from src.api import store
from src.api.schemas import ChatRequest


# ---- patch LLM ----
def _fake_llm_response(text: str):
    m = MagicMock()
    m.content = text
    return m


def patch_llms():
    """patch nodes 里的 get_llm / get_cheap_llm，不真调 API。"""
    from src.graph import nodes

    # generate 节点用 get_llm()：永远返回含原文不存在数字的假答案
    fake_answer = "公司去年营收为 9,999,999,999.99 元，[引用: 第1页]"

    def fake_get_llm():
        llm = MagicMock()
        llm.invoke.return_value = _fake_llm_response(fake_answer)
        return llm

    # route_query / grade_documents / rewrite_query 用 get_cheap_llm()：返回合理 JSON
    def fake_get_cheap_llm():
        llm = MagicMock()
        # route_query: direct/single/multi
        llm.invoke.return_value = _fake_llm_response(
            '{"route": "single", "standalone": "2025年营业收入", "reason": "test"}'
        )
        return llm

    nodes.get_llm = fake_get_llm
    nodes.get_cheap_llm = fake_get_cheap_llm
    print("[patch] get_llm -> fake answer with bogus number 9,999,999,999.99")
    print("[patch] get_cheap_llm -> fake JSON responses")


async def run_test():
    print("=== 启动 engine ===")
    engine.startup()
    patch_llms()

    sid = f"test-reset-{id(object())}"
    # 清掉这个 session 的历史（确保 fresh_session=True）
    store.load_history(sid)  # 确保存在

    req = ChatRequest(question="2025年营业收入是多少", session_id=sid)

    events = []
    cache_before = engine.cache_get(req.question)
    print(f"[setup] cache_before = {cache_before}")

    print("=== 跑 _chat_gen ===")
    from src.api.app import _chat_gen
    async for sse in _chat_gen(req):
        ev = sse["event"]
        data = json.loads(sse["data"])
        events.append((ev, data))
        if ev in ("reset", "done", "error"):
            print(f"  event: {ev} -> {str(data)[:120]}")

    # ---- 断言 ----
    from src.core.config import settings
    expected_resets = settings.max_regenerate + 1   # 1 次首答 + max_regenerate 次重生成
    resets = [d for ev, d in events if ev == "reset"]
    done_events = [d for ev, d in events if ev == "done"]
    errors = [d for ev, d in events if ev == "error"]

    print(f"\n=== 断言 ===")
    print(f"1. reset 事件数: {len(resets)} "
          f"(期望 {expected_resets} = max_regenerate({settings.max_regenerate}) + 1 首答)")
    assert len(resets) == expected_resets, \
        f"期望 {expected_resets} 次 reset（max_regenerate+1），实际 {len(resets)}"

    assert done_events, "没有 done 事件"
    done_answer = done_events[0]["answer"]
    print(f"2. done.answer = {done_answer[:80]}...")
    assert "数值无法在原文中核实" in done_answer, f"done.answer 不是 fallback 话术: {done_answer[:100]}"

    # 3. store 里存的是规范答案
    history = store.load_history(sid)
    print(f"3. store 历史: {len(history)} 轮, 最后一轮 answer = {history[-1]['a'][:80] if history else 'EMPTY'}...")
    assert history, "store 里没有历史"
    assert "数值无法在原文中核实" in history[-1]["a"], "store 里存的不是 fallback 话术"

    # 4. 缓存里没有写入
    cache_after = engine.cache_get(req.question)
    print(f"4. cache_after = {cache_after}")
    assert cache_after is None, f"缓存不该有写入，但有: {cache_after}"

    assert not errors, f"有 error 事件: {errors}"

    print("\n[OK] 全部断言通过")


if __name__ == "__main__":
    asyncio.run(run_test())
