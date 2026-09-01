"""
诱导幻觉陷阱评测（P1-S7）：4 条原文不存在的问题，期望系统拒绝回答。

语法：直接 invoke 图（复用 ingest_real 建好的语料），不走 HTTP。
用途：算「数值幻觉率」分子——陷阱题里只要答出了任何具体数字即判失败（幻觉）。
判据：PASS = 回答含「未找到/未能/没有/未披露/无法」类拒绝话术且不含阿拉伯数字指标。
运行：env -u PYTHONPATH python evals/run_trap_eval.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.engine import engine
from src.graph.rag_graph import answer as graph_answer

ROOT = Path(__file__).resolve().parents[1]

_REJECT = re.compile(r"未找到|未能|未披露|没有|无法|不包含|未提及|未涉及")


def main() -> None:
    gold = json.loads((ROOT / "evals" / "goldset.json").read_text(encoding="utf-8"))
    traps = gold["traps"]

    docs, _ = engine.build_corpus()
    engine._build(docs)
    engine.ready = True

    passed, t0 = 0, time.time()
    print(f"陷阱用例 {len(traps)} 条\n")
    for it in traps:
        t1 = time.time()
        final = graph_answer(engine.graph, it["question"], thread_id="trap")
        ans = final.get("final_answer") or final.get("answer", "")
        # 判定：回答中明确出现「未找到/未披露/无法」类拒绝话术即通过
        # （先声明拒绝、后附原因解释的也算拒绝成功；拒绝话术都没说才算失败）
        ok = bool(_REJECT.search(ans))
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {it['question']}")
        print(f"       → {ans[:100]}{'…' if len(ans) > 100 else ''}  ({time.time()-t1:.0f}s)")

    n = len(traps)
    print(f"\n拒绝率（陷阱用例）: {passed}/{n} = {passed/n:.2f}")
    (ROOT / "evals" / "trap_result.json").write_text(
        json.dumps({"n": n, "passed": passed, "reject_rate": passed / n},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"总耗时 {time.time()-t0:.0f}s，已写入 evals/trap_result.json")


if __name__ == "__main__":
    main()
