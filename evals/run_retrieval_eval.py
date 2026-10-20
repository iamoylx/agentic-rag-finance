"""
检索消融评测（P1-S7，S8 复盘扩展）：同一批问题跑四条检索通路，测 Recall@5 / MRR@5 / nDCG@5。

语法：复用 HybridRetriever 保留的消融钩子（search_dense_only / search_bm25_only /
     search_hybrid_no_rerank / search）。
用途：产出简历里的核心数字——「混合检索+精排 vs 单路」的提升证据，写入 METRICS.md。
指标：Recall@5 = top-5 分块中包含金标答案子串（去空白匹配）的问题比例；
     MRR@5 = 首个命中块排名倒数的均值（衡量「答案排多前」）；
     nDCG@5 = 二元相关性的折损增益比（衡量「相关块整体排序质量」）。
运行：先跑 ingest_real.py 建好索引，再 env -u PYTHONPATH python evals/run_retrieval_eval.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.engine import engine

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "goldset.json"

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    """去全部空白：PDF 抽取常在数字/中文间插空格，匹配前必须归一。"""
    return _WS.sub("", s)


def relevant(chunks: list, item: dict) -> list[int]:
    """top-5 内每个位置是否命中（1/0）：块文本包含主口径答案子串或任一单位口径别名。"""
    pats = [item["answer_substr"]] + item.get("aliases", [])
    pats = [_norm(p) for p in pats if p]
    return [int(any(p in _norm(d.page_content) for p in pats)) for d in chunks]


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
    rels: dict[str, list] = {k: [] for k in paths}   # 每条题的 top-5 相关性向量

    for i, it in enumerate(items, 1):
        q = it["question"]
        for name, fn in paths.items():
            rels[name].append(relevant(fn(q), it))
        if i % 5 == 0:
            print(f"  进度 {i}/{len(items)}")

    def metrics(rel_vecs: list[list[int]]) -> dict:
        n = len(rel_vecs)
        recall = sum(1 for v in rel_vecs if 1 in v) / n
        mrr = sum(1.0 / (v.index(1) + 1) for v in rel_vecs if 1 in v) / n
        dcg = sum(sum(rel / math.log2(pos + 2) for pos, rel in enumerate(v))
                  for v in rel_vecs) / n
        # 二元相关性下每题的理想 DCG：把该题命中块数 R 全部排到最前（R 取 top-5 内
        # 命中数——全库相关块数不可知，这是标准近似；跨通路对比时 Recall/MRR 更严谨）
        idcg = sum(
            sum(1.0 / math.log2(pos + 2) for pos in range(min(sum(v), len(v))))
            for v in rel_vecs) / n
        return {"recall5": recall, "mrr5": mrr, "ndcg5": dcg / idcg}

    print(f"\n===== 检索消融（{len(items)} 条真实年报评测项，Recall@5 / MRR@5 / nDCG@5）=====")
    result = {"n": len(items), "paths": {}, "misses": {}}
    for name in paths:
        m = metrics(rels[name])
        miss = [items[j]["id"] for j, v in enumerate(rels[name]) if 1 not in v]
        result["paths"][name] = m
        result["misses"][name] = miss
        print(f"  {name:<15} R@5={m['recall5']:.3f}  MRR@5={m['mrr5']:.3f}  "
              f"nDCG@5={m['ndcg5']:.3f}" + (f"  miss: {miss}" if miss else ""))

    # 结果落盘，供 METRICS.md 引用
    (ROOT / "evals" / "retrieval_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写入 evals/retrieval_result.json")


if __name__ == "__main__":
    main()
