"""
真实年报入库脚本（P1-S7 第一步）。

语法：复用 engine.build_corpus（解析+切块+缓存）与 engine._build（BM25+Chroma+精排就绪）。
用途：把 data/raw 下 3 份真实年报建成检索索引，打印解析统计供 METRICS.md 记录。
运行：env -u PYTHONPATH python scripts/ingest_real.py（需先停 8300 服务，避免双份模型占显存）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.engine import engine


def main() -> None:
    t0 = time.time()
    print("== 1/2 解析 + 切块（首次运行含表格抽取，约数分钟；之后走 corpus.jsonl 缓存）==")
    docs, stats = engine.build_corpus()
    for s in stats:
        print(f"  {s['file']}: {s['pages']}页 / {s['blocks']}块(表格{s['table_blocks']}) / {s['chunks']}分块")
    if not stats:
        print("  （命中缓存，跳过解析）")
    print(f"  总分块: {len(docs)}")

    print("== 2/2 建索引（BM25 + BGE-M3 入 Chroma）==")
    t1 = time.time()
    engine._build(docs)
    engine.ready = True
    print(f"  索引完成，耗时 {time.time() - t1:.1f}s")

    # 抽查：中文数字型问题应能命中真实年报
    hits = engine.retriever.search("贵州茅台2024年营业收入是多少", top_k=3)
    for d, s in hits:
        print(f"  [{s:.3f}] p{d.metadata.get('page')} {d.metadata.get('company')}: {d.page_content[:60]!r}")

    print(f"== 完成，总耗时 {time.time() - t0:.1f}s ==")


if __name__ == "__main__":
    main()
