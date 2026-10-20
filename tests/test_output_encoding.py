"""
运行时输出编码闸门（P1，2026-10-20）。

## 为什么需要它

Windows 控制台默认 cp936(GBK)。GBK **能**编码中文，却**不能**编码这些常用符号：

| 字符 | 码位 | GBK | 替代 |
|---|---|---|---|
| ✅ | U+2705 | 编不了 | `[OK]` |
| ❌ | U+274C | 编不了 | `[FAIL]` |
| ⚠ | U+26A0 | 编不了 | `[WARN]` |
| ¥ | U+00A5 | **编不了** | `￥`（U+FFE5，GBK 原生 `a3a4`，同形） |
| ⇒ | U+21D2 | 编不了 | `→`（U+2192，GBK 原生） |
| ↳ | U+21B3 | 编不了 | `→` |
| − | U+2212 | 编不了 | `-`（ASCII，或 `—` U+2014） |

后果不是乱码，是 **`UnicodeEncodeError` 直接把进程打死**——而且往往死在
「结论行」「成功提示」这种最需要被看见的位置。看起来像业务出错，实际是 IO 契约出错。

## 已发生的事故

- `scripts/verify_models.py`：两个模型都验证通过了，崩在最后一行打印 `✅ 全部通过` 上
- `scripts/smoke_test.py` / `test_hybrid.py` / `test_sse_reset.py`：验收脚本的结论行
- `evals/run_weight_sweep.py`：基线不符时**第一句警告**就带 `⚠️`
- `src/core/cost.py`：`BudgetExceeded` 与 `report()` 里的 `¥`（反直觉：以为 ¥ 肯定在 GBK 里）

## 为什么用 AST 而不是正则扫行

正则扫行会把**注释、文档字符串、以及产品文案和 prompt**里出现的符号也算成违规。
本仓库里就有三处必须**保留** emoji 的地方：

- `src/graph/nodes.py`：喂给 LLM 的**重生成提示**（改它等于改模型输入 → 改指标）
- `src/retrieval/embeddings.py`：给用户看的降级提示
- `final_answer` 里的「检测到回答中的数值无法在原文中核实」：**产品文案**

上一轮真用「整文件替换」扫过一次，正是把这三处改了——静态扫描必须只看
`Call` / `Raise` 节点，才能不碰它们。这就是本文件用 AST 而不是正则的原因。

纯静态扫描，不 import 被扫模块（避免副作用与慢加载）。零外部依赖。

## 本闸门的覆盖边界（诚实标注）

它只看 `print/sys.exit/logging.*/warnings.warn/argparse.error` 与 `raise` 两类节点。
**管不到「返回字符串、由调用方去打印」的格式化函数**，例如 `CostTracker.report()`——
它不是输出调用，静态扫描看不见它里面的 `¥`。

这是本轮实测出来的盲区：`test_cost_report_is_console_encodable` 一跑就红，
而主闸门是全绿的。处理方式不是把闸门放宽（那会让它开始误伤 prompt 与产品文案），
而是**给每个已知的格式化函数配一条运行时用例**：

| 格式化函数 | 对应运行时用例 |
|---|---|
| `CostTracker.report()` | `test_cost_report_is_console_encodable` |

以后新增「返回给人看的字符串」的函数，照这个模式补一条。
**静态闸门负责面，运行时用例负责点**——两者不能互相替代。
"""
import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCAN_DIRS = ("src", "scripts", "evals", "tests")
SKIP_PARTS = {".venv", "venv", "__pycache__", ".tmp", ".git", "node_modules"}

# 会把字符串写到 stdout/stderr 的函数名（取 Name.id 或 Attribute.attr，
# 所以 print / sys.exit / logging.warning / argparse.error 都会被认到）
OUTPUT_FUNCS = {
    "print", "exit", "error", "warn", "warning",
    "info", "debug", "critical", "exception", "write",
}

# Windows 控制台默认代码页；换机器若用其它 ANSI 代码页，这里跟着改
CONSOLE_ENCODING = "gbk"


def _py_files():
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            if SKIP_PARTS & set(p.parts):
                continue
            yield p


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _string_constants(node) -> list[ast.Constant]:
    """递归取出节点下所有字符串字面量（含 f-string 的固定片段）。"""
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _unencodable(text: str) -> list[tuple[str, str]]:
    bad = []
    for ch in text:
        if ord(ch) < 128:
            continue
        try:
            ch.encode(CONSOLE_ENCODING)
        except UnicodeEncodeError:
            bad.append((ch, f"U+{ord(ch):04X}"))
    return bad


def collect_offenders(paths) -> list[str]:
    """扫给定文件，返回所有「输出路径上含不可编码字符」的描述行。

    一次收集**全部**违规再返回，不在第一个违规处中断。
    """
    offenders: list[str] = []

    for path in paths:
        path = Path(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = path.name

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in OUTPUT_FUNCS:
                targets = list(node.args) + [kw.value for kw in node.keywords]
            elif isinstance(node, ast.Raise) and node.exc is not None:
                targets = [node.exc]
            else:
                continue

            for target in targets:
                for const in _string_constants(target):
                    bad = _unencodable(const.value)
                    if not bad:
                        continue
                    ch, cp = bad[0]
                    snippet = const.value.strip().replace("\n", " ")[:60]
                    offenders.append(f"  {rel}:{const.lineno}  {ch!r} {cp}  →  {snippet!r}")

    return offenders


# ---------------- 主闸门 ----------------
def test_output_paths_are_console_encodable():
    """所有「会打印出去」的字符串，都不许含 cp936 编不了的字符。

    覆盖两类出口，因为它们最终都到同一个控制台：
      · `ast.Call`  —— print / sys.exit / logging.* / argparse.error
      · `ast.Raise` —— 异常消息也会被打印（traceback，或 except 里 `str(e)`）
    """
    offenders = collect_offenders(_py_files())
    assert not offenders, (
        f"以下运行时输出含 {CONSOLE_ENCODING} 无法编码的字符，"
        f"在 Windows 控制台上会抛 UnicodeEncodeError（进程直接死，不是乱码）：\n"
        + "\n".join(offenders)
        + "\n\n改法：状态标记用 ASCII（[OK]/[WARN]/[FAIL]）；货币用 ￥(U+FFE5) 而非 ¥(U+00A5)；"
          "箭头用 → (U+2192) 而非 ⇒/↳。"
    )


def test_product_and_prompt_strings_may_keep_emoji(tmp_path):
    """**产品文案与 prompt 不在本闸门管辖内**——这是刻意划的边界。

    本轮改造时踩过：用「整文件替换」清扫 emoji，把 `final_answer` 的拦截文案、
    以及喂给 LLM 的重生成提示一起改了。后者**等于改模型输入**，会让 METRICS 里的
    指标失去可比性——那不是「编码修复」，是「行为变更」。

    所以闸门只认会到控制台的出口。这条用例把边界钉住：非输出调用里的 emoji 不报。
    """
    f = tmp_path / "product_text.py"
    f.write_text(
        'def answer():\n'
        '    return {"final_answer": "\u26a0\ufe0f 检测到数值无法核实，已拦截"}\n'
        '\n'
        'PROMPT_HINT = "\u2705 请重新作答"\n',
        encoding="utf-8",
    )
    assert collect_offenders([f]) == []


def test_scanner_ignores_comments_and_docstrings(tmp_path):
    """注释/docstring 里出现 `print("✅")` 不算违规——AST 相对正则的核心优势。

    真正的门是：**同样的文本放进注释不红、放进 print 才红**。两个方向都断言才算钉住。
    """
    prose_only = tmp_path / "prose_only.py"
    prose_only.write_text(
        '"""模块 docstring 里提到 print("\u2705") 是违规示例，但这里本身不是调用。"""\n'
        "# 注释里写 print('\u26a0 熔断') 也不该被判违规\n"
        "VALUE = 1\n",
        encoding="utf-8",
    )
    assert collect_offenders([prose_only]) == []

    real_call = tmp_path / "real_call.py"
    real_call.write_text('print("\u2705 全部通过")\n', encoding="utf-8")
    offenders = collect_offenders([real_call])
    assert len(offenders) == 1
    assert "U+2705" in offenders[0]
    assert "real_call.py:1" in offenders[0]


def test_scanner_catches_raise_messages_too(tmp_path):
    """异常消息里的 emoji 也要抓——`raise X("✅")` 比 print 更险。

    它平时不出现，只在出事时才现身，而「出事」正是最需要看清报错的时候。
    """
    f = tmp_path / "raiser.py"
    f.write_text('raise RuntimeError("\u2705 完成")\n', encoding="utf-8")
    offenders = collect_offenders([f])
    assert len(offenders) == 1 and "U+2705" in offenders[0]


# ---------------- 真实行为：成本报告必须能打到 GBK 控制台 ----------------
def test_cost_report_is_console_encodable():
    """`CostTracker.report()` 是给人看的收尾行，必须能在中文控制台打印。

    这条不只是重复主闸门：它走的是**真实 f-string 求值结果**，而不是源码里的字面量。
    若哪天 report() 改成从配置读货币符号，主闸门（静态扫源码）就看不见了，这条仍能抓。
    """
    from src.core.cost import CostTracker

    text = CostTracker().report()
    assert "实测成本" in text
    text.encode(CONSOLE_ENCODING)          # 编不了会抛，即失败
    assert "¥" not in text, "应使用 GBK 可编码的 ￥(U+FFE5)，而不是 ¥(U+00A5)"
