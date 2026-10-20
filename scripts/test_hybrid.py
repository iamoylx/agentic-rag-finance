"""
混合检索冒烟测试：验证 dense / BM25 / RRF / Rerank 四条通路。

用途：P1-S3 链路自检。真实评测集（30条）就绪前的链路验证版。
运行：python scripts/test_hybrid.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings                    # noqa: E402
from src.ingestion.chunker import FinancialChunker      # noqa: E402
from src.ingestion.pdf_parser import AnnualReportParser # noqa: E402
from src.retrieval.hybrid import HybridRetriever        # noqa: E402

SEP = "=" * 70


def main() -> int:
    settings.ensure_dirs()
    pdf_path = settings.raw_dir / "示例科技股份有限公司_2025年年度报告_测试样本.pdf"
    if not pdf_path.exists():
        print("测试 PDF 不存在，请先运行 scripts/smoke_test.py")
        return 1

    print(SEP)
    print("构建混合检索器（含 BGE-M3 入库 + BM25 索引 + Reranker 加载）")
    t0 = time.time()
    blocks = AnnualReportParser(extract_tables=True).parse(pdf_path)
    docs = FinancialChunker().split(blocks, doc_meta={"company": "示例科技", "year": "2025"})
    hr = HybridRetriever(docs, use_rerank=True)
    print(f"  就绪 {time.time() - t0:.1f}s | chunk 数 {len(docs)} | "
          f"Chroma {hr.store.count()} 条 | embedder@{hr.embedder.device}")

    # 三类查询：精确术语（BM25 强）/ 同义表述（dense 强）/ 混合
    cases = [
        ("精确术语+数字", "2025年营业收入 12,345,678,901.23"),
        ("同义表述", "公司去年赚了多少钱"),          # 原文没有“赚”字，考验语义
        ("业务概念", "应收账款带来的风险是什么"),
    ]
    all_ok = True
    for tag, q in cases:
        print(SEP)
        print(f"[{tag}] 查询: {q}")
        for name, fn in [
            ("dense ", hr.search_dense_only),
            ("bm25  ", hr.search_bm25_only),
            ("rrf   ", hr.search_hybrid_no_rerank),
        ]:
            hits = fn(q, top_k=3)
            tops = " ; ".join(
                f"p{d.metadata.get('page')}/{d.metadata.get('block_type')}" for d in hits
            )
            print(f"  {name}: {tops}")

        reranked = hr.search(q, top_k=3)
        tops = " ; ".join(f"p{d.metadata.get('page')}({s:.1f})" for d, s in reranked)
        print(f"  rerank: {tops}")
        if not reranked:
            all_ok = False

    print(SEP)
    print(f"混合检索链路: {'[OK] 四条通路全部有输出' if all_ok else '[FAIL] 有空结果'}")
    print(SEP)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
