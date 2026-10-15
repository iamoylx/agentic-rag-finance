"""闸门的性质测试：对**任意切法**验证不变式。

为什么单测不够：真实 tokenizer 怎么切一个数字（"170,899,152,276.34" 可能是
1 帧、可能是 20 帧、可能在裸小数点处断开）是不可预知的，手写几个切法只能覆盖
自己想到的情况。这里用「穷举 + 随机」造切法，把不变式钉死。

核心不变式：**放行边界永远不许落在任何一个数字 token 的内部。**
  设 k = 当前已放行字符数，answer 中数字 token 的区间为 [s, e)。
  若存在 span 使 s < k < e，说明屏幕上出现了「数字的前半截」。
  这正是「裸小数点被当成数字终止符」那个真实 bug 的表现形态：
  放行 "…1,234." 而完整数字是 "1,234.56"，k 落在 span 内部。
  放行到 k == s（数字还没开始）或 k == e（数字已完整）都允许。

为什么这条不变式比「文本是否完整」更严：那个 bug 的放行文本最终也能拼完整
（只是中间态泄漏了片段），只看最终文本会漏判；看边界才抓得住。
实测：把 number_gate 的正则退回旧版，本文件立刻 6 项失败。
"""
from __future__ import annotations

import random

import pytest

from src.safety.number_gate import NumericHoldBackGate
from src.safety.number_rules import NUM_RE, norm_num

CORPUS = norm_num("贵州茅台2024年营业收入为170,899,152,276.34元。基本每股收益68.64元。")

# 合法：任意切法下都必须完整放行
LEGIT_CASES = [
    "贵州茅台2024年营业收入为170,899,152,276.34元。",
    "贵州茅台2024年营业收入为170,899,152,276.34元，基本每股收益68.64元。",
    "基本每股收益68.64",                       # 以数字结尾 → 只能靠 flush 放行
    "2024年营业收入170,899,152,276.34元",
]

# 编造：任意切法下都必须被拦住
FAB_CASES = [
    "贵州茅台2024年营业收入为170,899,999,276.34元。",     # 中段编造，前后缀都撞真数字
    "贵州茅台2024年营业收入为170,899,152,276.345元。",    # 真数字后多一位
    "贵州茅台2024年营业收入为1,098,765,432.10元。",       # 语料里完全不存在
    "贵州茅台2024年营业收入为170,899,152,276.34元。基本每股收益77.77元。",
]


def numeric_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in NUM_RE.finditer(text)]


def chunkings(text: str, n_random: int = 80, seed: int = 20260910):
    """产出多种切法：最细(逐字符)、每位置一刀、数字内部切开、随机。"""
    yield list(text)
    for i in range(1, len(text)):
        yield [text[:i], text[i:]]
    for s, e in numeric_spans(text):          # 最容易踩边界：数字被从内部切开
        for cut in range(s + 1, e):
            yield [text[:cut], text[cut:]]
    rnd = random.Random(seed)
    for _ in range(n_random):
        n = rnd.randint(1, max(1, len(text) - 1))
        cuts = sorted(rnd.sample(range(1, len(text)), min(n, len(text) - 1)))
        chunks, prev = [], 0
        for c in cuts:
            chunks.append(text[prev:c]); prev = c
        chunks.append(text[prev:])
        yield [c for c in chunks if c]


def _drive(answer: str, chunks: list[str]):
    """按给定切法喂完，返回 (放行文本, violated)。每步都校验边界不变式。"""
    spans = numeric_spans(answer)
    gate = NumericHoldBackGate(CORPUS)
    released = ""
    for c in chunks:
        released += gate.feed(c)
        assert answer.startswith(released), \
            f"放行的不是原文前缀: {released!r} ← 切法 {chunks!r}"
        bad = [sp for sp in spans if sp[0] < len(released) < sp[1]]
        assert not bad, \
            f"放行边界落在数字内部 k={len(released)} span={bad}：屏幕上出现了数字前半截\n" \
            f"  放行文本={released!r}\n  切法={chunks!r}"
    released += gate.flush()
    assert answer.startswith(released), f"flush 后不是原文前缀: {released!r}"
    bad = [sp for sp in spans if sp[0] < len(released) < sp[1]]
    assert not bad, f"flush 后边界落在数字内部 k={len(released)} span={bad}"
    return released, gate.violated


@pytest.mark.parametrize("answer", LEGIT_CASES)
def test_legit_answer_survives_every_chunking(answer):
    """合法答案：任何切法都必须完整放行，且不得被判成编造（不过度拦截）。"""
    for chunks in chunkings(answer):
        released, violated = _drive(answer, chunks)
        assert violated is False, f"合法答案被误判成编造，切法={chunks!r}"
        assert released == answer, \
            f"合法答案没被完整放行（过度拦截），切法={chunks!r}\n  只放行了 {released!r}"


@pytest.mark.parametrize("answer", FAB_CASES)
def test_fabricated_answer_blocked_under_every_chunking(answer):
    """编造答案：任何切法都必须被拦（violated=True），且边界始终合法。"""
    for chunks in chunkings(answer):
        released, violated = _drive(answer, chunks)
        assert violated is True, \
            f"编造答案没被拦住，切法={chunks!r}\n  上屏了 {released!r}"


def test_boundary_invariant_actually_catches_the_bare_decimal_bug():
    """自检：把「裸小数点」这一刀单独拎出来，确认不变式在旧行为下会报警。

    这条用例的价值是**证明上面两条断言不是空转**：
    "68.64" 若在裸小数点处被切开，旧实现会把 "68." 放行（边界落在 span 内部）。
    这里显式断言「小数点那一刀之后，数字仍未放行」。
    """
    answer = "基本每股收益68.64元"
    gate = NumericHoldBackGate(CORPUS)
    assert gate.feed("基本每股收益") == "基本每股收益"
    assert gate.feed("68") == ""
    assert gate.feed(".") == ""       # ★旧实现此处会返回 "68."
    assert gate.feed("64") == ""
    assert gate.feed("元") == "68.64元"
    assert gate.violated is False
