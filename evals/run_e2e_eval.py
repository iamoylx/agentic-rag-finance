"""
端到端准确率评测（P1-S7 收尾）。

语法：逐条 invoke 完整图（含 LLM 生成与数值校验），按「答案是否包含金标值（含别名）」判对。
用途：产出简历指标「端到端准确率」；拒绝率指标由 run_trap_eval.py 负责，两者口径分离。
注意：会真实消耗 LLM 配额与余额；判对口径与检索评测一致（去空白 + 别名）。

成本可观测：每次 LLM 调用经 CostTracker 记账（含缓存命中/未命中分档），
          逐条打印累计成本，并支持 --cap 在超预算时**优雅中止**（不是崩在半路）。

运行：需先停 8300 服务（GPU 显存），env -u PYTHONPATH python evals/run_e2e_eval.py

  --limit N   只跑前 N 条（**试点版**，产物写 e2e_pilot.json，不碰正式的 e2e_result.json）
  --cap CNY   累计成本上限，超过则中止并保留已完成结果
  --out PATH  自定义产物路径
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.engine import engine
from src.core.cost import BudgetExceeded, CostTracker
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="只跑前 N 条做试点（产物默认写 e2e_pilot.json）")
    ap.add_argument("--cap", type=float, default=None,
                    help="累计成本上限（元），超出则中止并保留已完成结果")
    ap.add_argument("--out", type=str, default=None, help="产物路径")
    args = ap.parse_args()

    gold = json.loads((ROOT / "evals" / "goldset.json").read_text(encoding="utf-8"))
    items = [it for it in gold["items"] if it["kind"] in ("numeric", "qualitative")]
    if args.limit:
        items = items[:args.limit]

    # 试点与正式跑分开落盘：试点绝不能覆盖 README/METRICS 引用的正式结果
    out_path = Path(args.out) if args.out else (
        ROOT / "evals" / ("e2e_pilot.json" if args.limit else "e2e_result.json"))

    docs, _ = engine.build_corpus()
    engine._build(docs)
    engine.ready = True

    tracker = CostTracker(cap_cny=args.cap)
    print(f"模式：{'试点' if args.limit else '正式'}  n={len(items)}  "
          f"成本上限={'￥%.2f' % args.cap if args.cap else '未设'}")
    print(f"提供方：{engine.__class__.__name__} / 产物 → {out_path.relative_to(ROOT)}\n")

    results, passed, t0, aborted = [], 0, time.time(), False
    for i, it in enumerate(items, 1):
        t1, before = time.time(), tracker.cost_cny
        try:
            final = graph_answer(engine.graph, it["question"],
                                 thread_id=f"e2e-{i}", callbacks=[tracker])
            ans = final.get("final_answer") or final.get("answer", "")
            ok = correct(ans, it)
        except BudgetExceeded as e:
            print(f"\n★ 预算熔断，已中止：{e}")
            aborted = True
            break
        except Exception as e:
            ans, ok = f"ERROR {type(e).__name__}: {e}", False
        passed += ok
        results.append({"id": it["id"], "q": it["question"], "ok": ok, "answer": ans[:150]})
        print(f"[{'PASS' if ok else 'FAIL'}] {it['id']}  {ans[:70]}"
              f"  ({time.time()-t1:.0f}s, +￥{tracker.cost_cny-before:.4f})")

    n_done = len(results)
    acc = passed / n_done if n_done else 0.0
    print(f"\n===== {'（中止）' if aborted else ''}端到端准确率: "
          f"{passed}/{n_done} = {acc:.3f}  (目标 ≥0.75) =====")
    print(f"总耗时 {time.time()-t0:.0f}s")
    print(tracker.report())
    if n_done:
        print(f"单条均成本 ￥{tracker.cost_cny/n_done:.4f}"
              f"  → 按此推算 26 条约 ￥{tracker.cost_cny/n_done*26:.2f}")

    out_path.write_text(
        json.dumps({"n": n_done, "passed": passed, "accuracy": acc,
                    "aborted": aborted, "cost": tracker.as_dict(),
                    "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
