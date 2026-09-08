"""
端到端准确率评测（P1-S7 收尾）。

语法：逐条 invoke 完整图（含 LLM 生成与数值校验），按「答案是否包含金标值（含别名）」判对。
用途：产出简历指标「端到端准确率」；拒绝率指标由 run_trap_eval.py 负责，两者口径分离。
注意：会真实消耗 LLM 配额（26 条 × 3-5 次调用）；判对口径与检索评测一致（去空白 + 别名）。
运行：需先停 8300 服务（GPU 显存），env -u PYTHONPATH python evals/run_e2e_eval.py
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
_WS = re.compile(r"\s+")

# 正确答案允许出现的口径（主答案 + 别名，去空白/去 Markdown 标记后小写比较）
def _norm(s: str) -> str:
    return _WS.sub("", s.replace("*", "").replace("#", "")).lower()


def correct(ans: str, item: dict) -> bool:
    a = _norm(ans)
    return any(_norm(v) in a for v in [item["answer_substr"], *item.get("aliases", [])] if v)


def main() -> None:
    gold = json.loads((ROOT / "evals" / "goldset.json").read_text(encoding="utf-8"))
    items = [it for it in gold["items"] if it["kind"] in ("numeric", "qualitative")]

    docs, _ = engine.build_corpus()
    engine._build(docs)
    engine.ready = True

    results, passed, t0 = [], 0, time.time()
    for i, it in enumerate(items, 1):
        t1 = time.time()
        try:
            final = graph_answer(engine.graph, it["question"], thread_id=f"e2e-{i}")
            ans = final.get("final_answer") or final.get("answer", "")
            ok = correct(ans, it)
        except Exception as e:
            ans, ok = f"ERROR {type(e).__name__}: {e}", False
        passed += ok
        results.append({"id": it["id"], "q": it["question"], "ok": ok, "answer": ans[:150]})
        print(f"[{'PASS' if ok else 'FAIL'}] {it['id']}  {ans[:80]}  ({time.time()-t1:.0f}s)")

    n = len(items)
    acc = passed / n
    print(f"\n===== 端到端准确率: {passed}/{n} = {acc:.3f}  (目标 ≥0.75) =====")
    print(f"总耗时 {time.time()-t0:.0f}s")
    (ROOT / "evals" / "e2e_result.json").write_text(
        json.dumps({"n": n, "passed": passed, "accuracy": acc, "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入 evals/e2e_result.json")


if __name__ == "__main__":
    main()
