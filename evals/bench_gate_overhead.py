"""闸门性能实测：把「开销可忽略」「首字延迟不变」从说法变成数字。

三条要测的：
  A 单帧处理开销 —— 闸门 vs 无闸门基线（同一 token 流）
  B 首字延迟是否有变化 —— 时间上「第一个字符上屏」发生在喂到第几个 chunk
  C 被多扣的字符量 —— 扣留缓冲区的峰值长度（= 数字 token 长度）

答案文本取自 evals/e2e_result.json 的真实输出，token 流按 2 字符/块模拟
（贴近 AGNES/DeepSeek 的中文 token 粒度）。
零 LLM 消耗，纯本地 CPU。
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.safety.number_gate import NumericHoldBackGate          # noqa: E402
from src.safety.number_rules import NUM_RE, norm_num            # noqa: E402

# ---- 取真实答案 ----
res = json.loads((ROOT / "evals" / "e2e_result.json").read_text(encoding="utf-8"))["results"]
answers = [r.get("answer") or "" for r in res if r.get("answer")]
answers = [a for a in answers if len(a) > 40]
print(f"真实答案样本：{len(answers)} 条；"
      f"长度 中位 {statistics.median(len(a) for a in answers):.0f} / "
      f"最大 {max(len(a) for a in answers)}")

# 语料 = 所有答案里出现过的数字的并集（给闸门一个「足够宽松」的语料，
# 模拟真实场景：语料包含答案里的真数字，也包含别的数字）
CORPUS = norm_num("".join(answers))
print(f"语料长度 {len(CORPUS)} 字符")


def tokenize(text: str, size: int = 2) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def run_gate(text: str, n_rep: int = 30):
    """跑闸门，返回 (总耗时秒, 首字出现在第几个 chunk, 缓冲峰值, 放行字符数)。"""
    chunks = tokenize(text)
    t0 = time.perf_counter()
    first_at, peak = None, 0
    released = 0
    for _ in range(n_rep):
        g = NumericHoldBackGate(CORPUS)
        first_at, peak, released = None, 0, 0
        for i, c in enumerate(chunks):
            out = g.feed(c)
            peak = max(peak, len(g._buf))          # 诊断用：扣留缓冲长度
            if out:
                released += len(out)
                if first_at is None:
                    first_at = i
        out = g.flush()
        released += len(out)
        peak = max(peak, 0)
    dt = (time.perf_counter() - t0) / n_rep
    return dt, first_at, peak, released


def run_baseline(text: str, n_rep: int = 30):
    """无闸门基线：只做字符串拼接（模拟原实现逐 token 直发）。"""
    chunks = tokenize(text)
    t0 = time.perf_counter()
    first_at, released = None, 0
    for _ in range(n_rep):
        acc, first_at, released = "", None, 0
        for i, c in enumerate(chunks):
            acc += c
            released += len(c)
            if first_at is None:
                first_at = i
    dt = (time.perf_counter() - t0) / n_rep
    return dt, first_at, 0, released


print(f"\n{'—' * 76}")
print(f"{'答案长度':>8} {'基线 (ms)':>11} {'闸门 (ms)':>11} {'单帧开销 (μs)':>14} "
      f"{'首字@chunk':>12} {'缓冲峰值':>9}")
print(f"{'—' * 76}")

rows = []
for text in sorted(answers, key=len)[:6] + sorted(answers, key=len)[-3:]:
    b_dt, b_first, _, _ = run_baseline(text)
    g_dt, g_first, peak, released = run_gate(text)
    n_chunks = len(tokenize(text))
    per_frame = (g_dt - b_dt) / n_chunks * 1e6
    rows.append((len(text), b_dt, g_dt, per_frame, b_first, g_first, peak))
    print(f"{len(text):>8} {b_dt * 1e3:>11.3f} {g_dt * 1e3:>11.3f} {per_frame:>14.2f} "
          f"{str(b_first) + '→' + str(g_first):>12} {peak:>9}")

per = [r[3] for r in rows]
print(f"{'—' * 76}")
print(f"单帧开销：中位 {statistics.median(per):.2f} μs / 最大 {max(per):.2f} μs")
print(f"首字延迟：{'完全一致' if all(r[4] == r[5] for r in rows) else '★有差异'}"
      f"（基线 {[r[4] for r in rows][:5]} vs 闸门 {[r[5] for r in rows][:5]}）")
print(f"扣留缓冲峰值：中位 {statistics.median(r[6] for r in rows):.0f} / 最大 {max(r[6] for r in rows)} 字符")

# ---- 数字 token 长度分布：决定「被多扣多少」 ----
lens = []
for a in answers:
    lens += [m.end() - m.start() for m in NUM_RE.finditer(a)]
if lens:
    lens.sort()
    print(f"\n数字 token 长度：n={len(lens)}  中位 {statistics.median(lens):.0f}  "
          f"p90 {lens[int(len(lens) * 0.9)]}  最大 {max(lens)}")
    print(f"→ 即：闸门最多把 1 个数字（{max(lens)} 字符）多扣到拼完为止；"
          f"折算到 2 字符/块 = {max(lens) // 2} 个 chunk 的时长。")

# ---- 自检：闸门在真实答案上不该有任何编造判定（语料取自答案自身） ----
bad = []
for a in answers:
    g = NumericHoldBackGate(CORPUS)
    out = "".join(g.feed(c) for c in tokenize(a)) + g.flush()
    if g.violated or out != a:
        bad.append((a[:40], g.violated, out == a))
print(f"\n自检：{len(answers)} 条真实答案过闸门，异常 {len(bad)} 条 "
      f"{'（语料含全部答案数字，故应为 0）' if not bad else bad[:3]}")
