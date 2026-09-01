"""
检索消融评测（P1-S7）：同一批问题跑四条检索通路，测 Recall@5。

语法：复用 HybridRetriever 保留的消融钩子（search_dense_only / search_bm25_only /
     search_hybrid_no_rerank / search）。
用途：产出简历里的核心数字——「混合检索+精排 vs 单路」的提升证据，写入 METRICS.md。
指标：Recall@5 = top-5 分块中包含金标答案子串（去空白匹配）的问题比例。
运行：先跑 ingest_real.py 建好索引，再 env -u PYTHONPATH python evals/run_retrieval_eval.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.engine import engine
from src.retrieval.reranker import BGEReranker

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "goldset.json"

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    """去全部空白：PDF 抽取常在数字/中文间插空格，匹配前必须归一。"""
    return _WS.sub("", s)


def hit(chunks: list, item: dict) -> bool:
    """命中判定：块文本包含主口径答案子串或任一单位口径别名（全部去空白匹配）。"""
    pats = [item["answer_substr"]] + item.get("aliases", [])
    pats = [_norm(p) for p in pats if p]
    return any(any(p in _norm(d.page_content) for p in pats) for d in chunks)


def main() -> None:
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    items = [it for it in gold["items"] if it["kind"] in ("numeric", "qualitative")]
    print(f"评测集: {len(items)} 条检索项")

    docs, stats = engine.build_corpus()
    print(f"语料: {len(docs)} 分块，建索引中…")
    t0 = time.time()
    engine._build(docs)          # 含 reranker（完整通路）
    print(f"索引就绪 ({time.time() - t0:.0f}s)\n")

    r = engine.retriever
    paths = {
        "dense_only":      lambda q: [d for d in r.search_dense_only(q, top_k=5)],
        "bm25_only":       lambda q: [d for d, _ in r.bm25.search(q, top_k=5)],
        "hybrid(RRF)":     lambda q: r.search_hybrid_no_rerank(q, top_k=5),
        "hybrid+rerank":   lambda q: [d for d, _ in r.search(q, top_k=5)],
    }
    scores = {k: 0 for k in paths}
    misses: dict[str, list] = {k: [] for k in paths}

    for i, it in enumerate(items, 1):
        q = it["question"]
        for name, fn in paths.items():
            if hit(fn(q), it):
                scores[name] += 1
            else:
                misses[name].append(it["id"])
        if i % 5 == 0:
            print(f"  进度 {i}/{len(items)}")

    print("\n===== Recall@5（26 条真实年报评测项）=====")
    for name in paths:
        print(f"  {name:<15} {scores[name]}/{len(items)} = {scores[name]/len(items):.3f}"
              + (f"  miss: {misses[name]}" if misses[name] else ""))

    # 结果落盘，供 METRICS.md 引用
    result = {"n": len(items), "recall5": {k: scores[k] / len(items) for k in paths},
              "misses": misses}
    (ROOT / "evals" / "retrieval_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写入 evals/retrieval_result.json")


if __name__ == "__main__":
    main()
