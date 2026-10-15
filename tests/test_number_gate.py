"""闸门与数字规则的单元测试（不启 HTTP，直接测纯逻辑）。

为什么要单独有一组：SSE 测试（test_sse_gate.py）验证的是「接线接对了」，
而闸门的 bug 绝大多数藏在**token 边界**上——数字被模型拆成好几个 token，
扣留与放行的时机差一个字符，行为就完全不同。这类边界用纯函数测最省事、最全。
"""
from __future__ import annotations

from src.safety.number_gate import NumericHoldBackGate
from src.safety.number_rules import (
    build_corpus_norm, meaningful_numbers, missing_numbers, norm_num,
)

CORPUS = norm_num("贵州茅台2024年营业收入为170,899,152,276.34元。基本每股收益68.64元/股。")


# =====================================================================
# 数字规则
# =====================================================================

def test_norm_num_strips_thousand_separators():
    assert norm_num("170,899,152,276.34") == "170899152276.34"
    assert norm_num("1，234") == "1234"          # 全角逗号
    assert norm_num("68.64.") == "68.64"        # 尾部裸小数点


def test_meaningful_numbers_filters_page_and_index_noise():
    # "5"（页码）与 "1"（序号）长度不足 2 → 不计入
    assert meaningful_numbers("见第5页，共1项") == set()
    # 4 位年份要管：它就是数字，也必须能在原文里找到
    assert "2024" in meaningful_numbers("贵州茅台2024年")


def test_missing_numbers_detects_fabrication():
    assert missing_numbers("营业收入为170,899,152,276.34元", CORPUS) == []
    assert missing_numbers("营业收入为999,888,000元", CORPUS) == ["999888000"]


def test_build_corpus_norm_joins_docs():
    class D:
        def __init__(self, t): self.page_content = t
    assert build_corpus_norm([D("1,234"), D("5,678")]) == "1234\n5678"


def test_check_hallucination_node_still_uses_shared_rules():
    """回归：数字规则被抽到 number_rules 后，图节点判定必须与纯函数一致。"""
    from src.graph.nodes import check_hallucination

    class D:
        def __init__(self, t): self.page_content = t

    ok = check_hallucination({"answer": "营收170,899,152,276.34元",
                              "documents": [D("营收为170,899,152,276.34元")]})
    assert ok["hallucination_passed"] is True

    bad = check_hallucination({"answer": "营收999,888,000元",
                               "documents": [D("营收为170,899,152,276.34元")]})
    assert bad["hallucination_passed"] is False
    assert "999888000" in bad["trace"][0]


# =====================================================================
# 闸门：token 边界
# =====================================================================

def _feed_all(gate, chunks):
    return "".join(gate.feed(c) for c in chunks) + gate.flush()


def test_holds_number_split_across_many_tokens():
    """模型把 170,899,152,276.34 拆成 4 个 token → 拼齐前一个字符都不许上屏。"""
    gate = NumericHoldBackGate(CORPUS)
    out = []
    out.append(gate.feed("营业收入为"))
    for c in ("170,", "899,152,", "276.34"):
        assert gate.feed(c) == "", f"数字还没拼完就放行了: {c!r}"
    out.append(gate.feed("元。"))
    shown = "".join(out)
    assert shown == "营业收入为170,899,152,276.34元。"
    assert gate.violated is False


def test_fabricated_number_is_withheld_and_flagged():
    gate = NumericHoldBackGate(CORPUS)
    assert gate.feed("营业收入为") == "营业收入为"     # 正文照常放行
    assert gate.feed("999") == ""
    assert gate.feed("元。") == ""                    # 判定失败 → 不放行
    assert gate.violated is True
    assert gate.flush() == ""                         # violated 后不再放行
    assert gate.feed("后面还有内容") == ""


def test_trailing_number_released_by_flush():
    """答案以数字结尾：没有后续非数字字符来触发判定，必须靠 flush 放行。"""
    gate = NumericHoldBackGate(CORPUS)
    assert gate.feed("每股收益为") == "每股收益为"
    assert gate.feed("68.64") == ""                   # 扣住
    assert gate.flush() == "68.64"                    # 放行


def test_flush_also_verifies():
    """flush 不能只放行不判定，否则末尾的编造数字会从后门溜出去。"""
    gate = NumericHoldBackGate(CORPUS)
    gate.feed("每股收益为")
    gate.feed("77.77")
    assert gate.flush() == ""
    assert gate.violated is True


def test_full_width_comma_number_held_correctly():
    """全角千分位同样要扣住——正则若漏了「，」就会在逗号处误判数字已结束。

    注意语料里必须真的存在这个数字：闸门对「语料中没有的数字」一定会拒绝放行
    （这正是它的职责）。第一版这个用例把 1，234 写进了断言却没写进语料，
    结果被闸门正确处理成「编造」→ 测试红。是测试的错，不是实现的错。
    """
    corpus = norm_num("营业收入为1，234元。")
    gate = NumericHoldBackGate(corpus)
    assert gate.feed("营收") == "营收"
    assert gate.feed("1，234") == ""          # 逗号处不许当数字已结束
    assert gate.feed("元。") == "1，234元。"   # 遇到非数字字符才判定并放行
    assert gate.violated is False


def test_year_is_verified_not_held_forever():
    """"2024年" 里 2024 是有效数字，但后面跟了非数字字符 → 立刻判定并放行。"""
    gate = NumericHoldBackGate(CORPUS)
    assert gate.feed("贵州茅台") == "贵州茅台"
    assert gate.feed("2024年") == "2024年"
    assert gate.violated is False
