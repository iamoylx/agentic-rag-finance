"""闸门的性质测试：把「屏幕上的数字都是校验过的」这条对外承诺钉死。

## 覆盖边界（诚实版，别对外说成「覆盖了所有切法」）

全部切法是 `2^(n-1)` 种；真实答案长度中位数 51 字 ≈ 5.6e14 种，物理上不可能穷举。
本文件用三层拼出结论，**每层承担的责任不同**：

1. **可证**（不靠跑）：任意切法下「放行边界不落在任何数字 token 内部」都成立——
   放行只发生在两种情形，各自矛盾：
     · 整体放行时，若已喂文本正停在某数字 token 内部，则该 token 首位到结尾全由
       「数字 / 千分位逗号 / 一个小数点」组成，必然匹配尾部正则 → 会被扣住。
       故「不匹配」⟹ 没停在数字内部。
     · 部分放行时，正则取**最左**匹配位置 m（正则锚在 `$`，匹配必为后缀）。
       若 m 落在某 token [s,e) 内（s<m），则从 s 到结尾同样是「数字/逗号/+一个点」，
       而 `[\\d,，]*` 能吸收逗号、`\\.\\d*` 能吸收小数点 → 正则也能从 s 匹配，
       与「m 最左」矛盾。
   （论证依赖：NUM_RE 与尾部正则都最多含一个小数点，故不存在「两个点」打破论证的构造。）
2. **穷举小实例**：短答案上跑遍全部 `2^(n-1)` 种切法（见 test_exhaustive_short_answers）。
3. **系统采样作回归护栏**：长答案上跑逐字符 + 所有单刀 + 数字内部 ≥2 刀 + 随机多刀
   （见 chunkings()）。采样只承担「防止未来改动引入回归」，**不承担证明责任**。

## 四条不变式（全部在 _drive 里逐步断言）

- I1 放行文本始终是原文前缀；
- I2 **放行边界不落在任何数字 token 内部**（这条抓住了「裸小数点」那个真 bug）；
- I3 **屏幕上出现过的任何数字都必须能在语料里找到**（项目对外承诺的那句话）；
- I4 判定（violated）与切法无关（见 test_gate_verdict_is_chunking_independent），
  且与权威校验节点一致（见 test_gate_verdict_matches_authoritative_check）。
  注意是「**判定**与切法无关」，不是「放行文本相等」——后者是错的，见该用例 docstring。

## 各条断言的分工（变异验证实测，别混为一谈）

断言写在测试里，但**它能不能真的探测到缺陷**是另一回事。下面这张表是把被测代码
按「只改一件事」的方式破坏掉之后，实测每条断言各自的触发次数（10 个用例）：

| 变异（只改这一件事） | I1 前缀 | I2 边界 | I3 屏幕数字 | I4a 判定一致 |
|---|---|---|---|---|
| 基线（不破坏） | 0 | 0 | 0 | 0 |
| M1 旧正则 `\\.\\d+`（真实发生过的 bug） | 0 | **7** | **0** | **0** |
| M2 判定脱钩（放行序列逐字节不变，只把结论改成 False） | 0 | 0 | 0 | **5** |
| M3 闸门旁路（所有内容原样上屏） | 0 | **10** | **5** | **5** |
| M4 放行路径清洗（`safe.replace(",", "")`） | **5** | 5 | 0 | 0 |

读法（这张表的信息量比「全绿」大得多）：

- **每条断言都至少被一种变异触发** → 没有空转（恒真的）断言。
- **M1 只有 I2 能抓**：旧正则的缺陷是**显示时序**问题——最终文本其实拼完整了，
  只有中途某一帧把 `"170,899,152,276."` 放上了屏。所以：
  · I3 抓不住。`missing_numbers("…276.")` 归一化后是 `170899152276`，它**确实**
    是语料里 `170899152276.34` 的子串 → 判定通过。要抓它必须看「边界」，
    而不是看「数字在不在语料里」。
  · I4a 也抓不住，而且**本该抓不住**：此时闸门 violated 与权威 passed 完全一致
    （两者都认为整段话有问题）。缺陷不在结论上，在屏幕上。
- **M2 只有 I4a 能抓**：这正是 I4a 不可被替代的证据——放行序列一个字节都没变，
  其它三条断言全部沉默。
- **I1 目前只在 M4 下响**：它防的是「放行不再连续」这一类未来改动
  （比如有人为了美观在释放路径上加清洗/重排），是结构性护栏，不是主探测器。
- **M3（闸门整个失效）四条全响**——最粗的破坏被最多的断言拦住，符合预期；
  但反过来说，只测「闸门开/关」是**最弱的证据**，因为它区分不出细粒度缺陷。

上述 M1–M4 的复现脚本见 `.tmp/three_mutations.py`（临时脚本，不入库）。

## 为什么要有 I4 这两条差分测试

闸门与 `check_hallucination` 共用 `missing_numbers`（见 src/safety/number_rules.py 的模块注释）。
这个契约一旦破了，会出现最诡异的现象：**屏幕完整放行了内容，最后却弹「校验失败，触发重生成」**
——用户会觉得系统疯了。差分测试就是守这个契约的。
并且「结论与切法无关」是采样能替代穷举的**前提**：若结论依赖切法，采样再多也不等于覆盖。
"""
from __future__ import annotations

import random

import pytest

from src.safety.number_gate import NumericHoldBackGate
from src.safety.number_rules import NUM_RE, missing_numbers, norm_num

RAW_CORPUS = "贵州茅台2024年营业收入为170,899,152,276.34元。基本每股收益68.64元。"
CORPUS = norm_num(RAW_CORPUS)

# 合法：任何切法下都必须完整放行
LEGIT_CASES = [
    "贵州茅台2024年营业收入为170,899,152,276.34元。",
    "贵州茅台2024年营业收入为170,899,152,276.34元，基本每股收益68.64元。",
    "基本每股收益68.64",                        # 以数字结尾 → 只能靠 flush 放行
    "2024年营业收入170,899,152,276.34元",
    "2024 年营业收入 170,899,152,276.34 元",     # 含空白：数字与空白相邻的边界
]

# 编造：任何切法下都必须被拦住
FAB_CASES = [
    "贵州茅台2024年营业收入为170,899,999,276.34元。",     # 中段编造，前后缀都撞真数字
    "贵州茅台2024年营业收入为170,899,152,276.345元。",    # 真数字后多一位
    "贵州茅台2024年营业收入为1,098,765,432.10元。",       # 语料里完全不存在
    "贵州茅台2024年营业收入为170,899,152,276.34元。基本每股收益77.77元。",
    "2024 年营业收入 1,098,765,432.10 元",                 # 含空白 + 编造
]

# 短答案：可全量穷举 2^(n-1) 的（答案, 是否该被拦）
# 注意期望值是对**本文件的 CORPUS** 而言：CORPUS 里只有 170,899,152,276.34 与 68.64，
# 所以 "1.2" / "1,234.56" / "170,899,152,276.345" 都算编造。
EXHAUSTIVE_CASES = [
    ("营收1.2元", True),             # 裸小数点 + 语料里没有 1.2 → 编造，必须拦
    ("收益6.6元", True),             # 编造，前缀撞真实 68.64
    ("68.64", False),                # 纯数字收尾，靠 flush（合法，且踩裸小数点边界）
    ("收益68.64元", False),          # 合法 + 裸小数点边界
    ("1,234.56元", True),            # 前缀撞车 + 多半位
    ("170,899,152,276.345", True),   # n=19 → 262144 种，最刁钻
]


def numeric_spans(text: str) -> list[tuple[int, int]]:
    """答案里每个数字 token 的区间 [start, end)。"""
    return [(m.start(), m.end()) for m in NUM_RE.finditer(text)]


def chunkings(text: str, n_random: int = 80, seed: int = 20260910):
    """产出多种切法（**都是原文本的一个划分**，即 join 后等于原文）。

    四个族：
      ① 逐字符                      —— 1 种
      ② 每个位置单刀                —— n-1 种（对「单刀」是穷举）
      ③ 同一数字内部切 ≥2 刀         —— 把数字切成 3 段以上；这是 ② 的**补集**
                                      （单刀只能切成 2 段，此前这一形态零覆盖）
      ④ 随机多刀（固定种子，可复现）  —— 80 种，并随机插入空串模拟网关空 delta
    另含「空白字符单独成块」，用于数字与空白相邻的终止符判定。
    """
    yield list(text)

    for i in range(1, len(text)):                       # ②
        yield [text[:i], text[i:]]

    for s, e in numeric_spans(text):                    # ③ 数字内部 ≥2 刀
        inner = list(range(s + 1, e))
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                a, b = inner[i], inner[j]
                yield [text[:a], text[a:b], text[b:]]

    for i, ch in enumerate(text):                       # 空白单独成块
        if ch.isspace():
            yield [c for c in (text[:i], text[i], text[i + 1:]) if c]

    rnd = random.Random(seed)                           # ④ 随机多刀
    for _ in range(n_random):
        n = rnd.randint(1, max(1, len(text) - 1))
        cuts = sorted(rnd.sample(range(1, len(text)), min(n, len(text) - 1)))
        chunks, prev = [], 0
        for c in cuts:
            chunks.append(text[prev:c]); prev = c
        chunks.append(text[prev:])
        chunks = [c for c in chunks if c]
        if rnd.random() < 0.3:                          # 空串：网关可能吐空 delta
            chunks.insert(rnd.randrange(len(chunks) + 1), "")
        yield chunks


def _drive_raw(answer: str, chunks: list[str]) -> tuple[str, bool]:
    """按给定切法喂完，**不做任何断言**，返回 (累计放行文本, violated)。

    为什么要有这个「不设防」的版本：差分测试（I4a/I4b）问的是
    「闸门**结论**是否与权威校验一致」，与「放行**边界**是否合法」是两件事。
    若差分测试走断言版，一旦 I2 先炸就先抛异常、I4a 的断言根本执行不到
    ——测试报告的失败原因就名不副实了。实测：退回旧正则时只有 I2 报警、
    I4a 其实完全一致（那个缺陷是**显示时序**问题，不是**结论**问题）。
    """
    assert "".join(chunks) == answer, f"切法不是原文本的一个划分: {chunks!r}"
    gate = NumericHoldBackGate(CORPUS)
    released = ""
    for c in chunks:
        released += gate.feed(c)
    released += gate.flush()
    return released, gate.violated


def _drive(answer: str, chunks: list[str]) -> tuple[str, bool]:
    """按给定切法喂完，逐步断言 I1/I2/I3，返回 (累计放行文本, violated)。

    「每一步」都要断言：缺陷恰恰可能只出现在中间某一帧（旧正则就是这样——
    最终文本其实是完整的，只有中途那一帧把 "170,899,152,276." 放上了屏）。
    只查最终状态会漏掉它。
    """
    assert "".join(chunks) == answer, f"切法不是原文本的一个划分: {chunks!r}"
    spans = numeric_spans(answer)
    gate = NumericHoldBackGate(CORPUS)
    released = ""

    for c in chunks:
        released += gate.feed(c)
        _assert_invariants(answer, released, spans, chunks)

    released += gate.flush()
    _assert_invariants(answer, released, spans, chunks, stage="flush")
    return released, gate.violated


def _assert_invariants(answer, released, spans, chunks, stage="feed"):
    # I1：屏幕上的一切都必须是原文的前缀（否则就是凭空造字）
    assert answer.startswith(released), \
        f"[{stage}] 放行的不是原文前缀: {released!r} ← 切法 {chunks!r}"

    # I2：放行边界不许落在数字 token 内部
    #     （k==span 起点「数字还没开始」或 k==span 终点「数字已完整」都合法）
    bad = [sp for sp in spans if sp[0] < len(released) < sp[1]]
    assert not bad, (
        f"[{stage}] 放行边界落在数字内部 k={len(released)} span={bad}："
        f"屏幕上出现了数字前半截\n  放行文本={released!r}\n  切法={chunks!r}")

    # I3：屏幕上出现过的任何数字，都必须能在语料里找到
    #     —— 这条就是项目对外承诺的那句话（「数字都能在原文里核实」）
    miss = missing_numbers(released, CORPUS)
    assert not miss, (
        f"[{stage}] 屏幕上出现了语料里没有的数字 {miss}"
        f"\n  放行文本={released!r}\n  切法={chunks!r}")


def _authority_passes(text: str) -> bool:
    """权威校验节点对同一文本的结论（闸门的结论必须与它一致）。"""
    from src.graph.nodes import check_hallucination

    class _D:
        page_content = RAW_CORPUS

    return check_hallucination({"answer": text, "documents": [_D()]})["hallucination_passed"]


# =====================================================================
# 基本性质：合法必放行 / 编造必拦
# =====================================================================

@pytest.mark.parametrize("answer", LEGIT_CASES)
def test_legit_answer_fully_released_under_generated_chunkings(answer):
    """合法答案：任何生成的切法都必须完整放行，且不得被判成编造（不过度拦截）。"""
    for chunks in chunkings(answer):
        released, violated = _drive(answer, chunks)
        assert violated is False, f"合法答案被误判成编造，切法={chunks!r}"
        assert released == answer, \
            f"合法答案没被完整放行（过度拦截），切法={chunks!r}\n  只放行了 {released!r}"


@pytest.mark.parametrize("answer", FAB_CASES)
def test_fabricated_answer_blocked_under_generated_chunkings(answer):
    """编造答案：任何生成的切法都必须被拦（violated=True），且边界始终合法。

    注意用例名：是 generated（生成的采样），不是 every（全部）——别 overclaim。
    """
    for chunks in chunkings(answer):
        released, violated = _drive(answer, chunks)
        assert violated is True, \
            f"编造答案没被拦住，切法={chunks!r}\n  上屏了 {released!r}"


# =====================================================================
# 差分一致性：整个设计的地基
# =====================================================================

@pytest.mark.parametrize("answer", LEGIT_CASES + FAB_CASES)
def test_gate_verdict_matches_authoritative_check(answer):
    """闸门结论 ⟺ 权威校验节点结论（对同一文本）。

    为什么必须有：两者共用 missing_numbers，一旦漂移就会出现
    「屏幕完整放行了，最后却弹校验失败、触发重生成」——用户会觉得系统疯了。
    这正是 src/safety/number_rules.py 模块注释里警告的后果。

    注意用 _drive_raw：本用例只关心**结论**是否一致，不关心边界是否合法
    （边界由上面的用例负责）。否则 I2 会先抛异常，让本用例以错误的名义失败。
    实测 M2「判定脱钩」变异（只把结论改成 False、放行序列逐字节不变）下，
    本用例是**唯一**报警的断言 → 它有独立灵敏度，不是重复。
    """
    authoritative = _authority_passes(answer)
    for chunks in chunkings(answer):
        _, violated = _drive_raw(answer, chunks)
        assert (not violated) == authoritative, (
            f"闸门与权威校验结论不一致：闸门 violated={violated}，"
            f"权威 passed={authoritative}，切法={chunks!r}")


@pytest.mark.parametrize("answer", LEGIT_CASES + FAB_CASES)
def test_gate_verdict_is_chunking_independent(answer):
    """闸门的**判定**（violated）不许随切法变化。

    这是采样能替代穷举的前提：权威校验只看整段文本、天然与切法无关；
    若闸门的判定依赖 tokenizer 怎么切，就会出现「同一句话换个模型/换个网关，
    屏幕表现不一样」——不可接受。反之，只要判定与切法无关，
    「所有单刀（穷举）+ 多刀采样（护栏）」就等价于覆盖了全部切法。

    ★ 不断言「放行文本相等」——我最初就是这么断的，测试跑红了才发现是错的：
      同一编造答案在不同切法下有 11 种不同的放行结果，例如
        单块切法 [T]            → 放行 ''
        逐字符切法 ['贵','州',…] → 放行 '贵州茅台2024年'
      三种都 violated=True，但放行量不同。原因是粗粒度切法把「数字前那段正文」
      和数字放进了同一个 safe 块，于是**一起**被拒；细粒度切法在前几帧就已把
      那段正文放行了。**粗粒度只会更保守，不会更激进**，所以这不是缺陷。
      真正需要"放行量"维度的保证是 I2（放行边界）——它由 _drive 在每个切法上断言，
      也是唯一能抓住「裸小数点」那个 bug 的断言。
    """
    verdicts = {_drive_raw(answer, c)[1] for c in chunkings(answer)}
    assert len(verdicts) == 1, (
        f"同一答案在不同切法下判定不一致：{verdicts}（期望只有一个值）")


# =====================================================================
# 补充边界：空串分块、短答案全穷举
# =====================================================================

@pytest.mark.parametrize("answer", [LEGIT_CASES[0], FAB_CASES[0]])
def test_empty_chunk_is_a_noop(answer):
    """空串分块必须是彻底的 no-op——真实网关/适配层可能吐空 delta。

    此前零覆盖（没人测过 feed("")）。做法：在逐字符切法的**每个位置**插入空串，
    断言结论与不插入时逐字一致。
    """
    base = list(answer)
    baseline = _drive(answer, base)
    for pos in range(len(base) + 1):
        injected = base[:pos] + [""] + base[pos:]
        assert _drive(answer, injected) == baseline, f"在第 {pos} 位插入空串改变了结论"


@pytest.mark.parametrize("answer,should_block", EXHAUSTIVE_CASES)
def test_exhaustive_short_answers(answer, should_block):
    """短答案：跑遍**全部** 2^(n-1) 种切法（不是采样）。

    这是「所有切法」这个强主张的可执行证据层；长答案靠模块 docstring 里的论证 +
    采样护栏。最大一组 n=19 → 262144 种。
    """
    spans = numeric_spans(answer)
    n = len(answer)
    seen = 0
    for mask in range(1 << (n - 1)):
        cuts = [i + 1 for i in range(n - 1) if mask >> i & 1]
        chunks, prev = [], 0
        for c in cuts:
            chunks.append(answer[prev:c]); prev = c
        chunks.append(answer[prev:])
        chunks = [c for c in chunks if c]
        seen += 1
        gate = NumericHoldBackGate(CORPUS)
        released = ""
        for c in chunks:
            released += gate.feed(c)
            _assert_invariants(answer, released, spans, chunks)
        released += gate.flush()
        _assert_invariants(answer, released, spans, chunks, stage="flush")
        assert gate.violated is should_block, (
            f"穷举到切法 {chunks!r} 时结论不符：violated={gate.violated} 期望 {should_block}")
    assert seen == 1 << (n - 1)


def test_boundary_invariant_actually_catches_the_bare_decimal_bug():
    """自检：把「裸小数点」这一刀单独拎出来，确认 I2 不是空转。

    "68.64" 若在裸小数点处被切开，旧实现会把 "68." 放行（边界落在 span 内部）。
    """
    answer = "基本每股收益68.64元"
    gate = NumericHoldBackGate(CORPUS)
    # 逐段喂入后把各段结果收集成列表再比较：串行 assert 会互相遮蔽
    # （第一条红了下文就不跑），列表比较一次给出全部差异。
    parts = [gate.feed("基本每股收益"), gate.feed("68"), gate.feed("."),
             gate.feed("64"), gate.feed("元")]
    assert parts == ["基本每股收益", "", "", "", "68.64元"]   # ★旧实现在第三段会返回 "68."
    # 「无损」的定义：各段放行内容按原序拼回去，必须与原文逐字相同
    assert "".join(parts) == answer
    assert gate.violated is False
