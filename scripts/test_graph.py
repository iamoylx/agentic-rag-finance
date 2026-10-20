"""
LangGraph 状态图行为验证：路由 / 检索 / 直答 / 改写 / 降级。

用途：P1-S4 验收。对应方案中的行为用例表（4 条核心路径）。
运行：python scripts/test_graph.py ["可选：自定义问题"]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings                     # noqa: E402
from src.ingestion.chunker import FinancialChunker       # noqa: E402
from src.ingestion.pdf_parser import AnnualReportParser  # noqa: E402
from src.graph.rag_graph import answer, build_rag_graph  # noqa: E402
from src.retrieval.hybrid import HybridRetriever         # noqa: E402

SEP = "=" * 70


def main() -> int:
    settings.ensure_dirs()
    pdf_path = settings.raw_dir / "示例科技股份有限公司_2025年年度报告_测试样本.pdf"
    if not pdf_path.exists():
        print("请先运行 scripts/smoke_test.py 生成测试 PDF")
        return 1

    print("构建图（检索索引 + LangGraph 编译）...")
    blocks = AnnualReportParser(extract_tables=True).parse(pdf_path)
    docs = FinancialChunker().split(blocks, doc_meta={"company": "示例科技", "year": "2025"})
    graph = build_rag_graph(HybridRetriever(docs, use_rerank=True))
    print("就绪。\n")

    cases = [
        ("概念直答（期望 route=direct，不检索）", "什么是商誉？"),
        ("事实检索（期望命中第2页，带引用）", "公司2025年营业收入是多少？"),
        ("口语化（期望改写为财报术语再检索）", "公司去年赚了多少钱？"),
        ("原文没有的问题（期望优雅降级）", "公司2025年在中国境外的收入占比是多少？"),
    ]
    custom = sys.argv[1] if len(sys.argv) > 1 else None
    if custom:
        cases = [("自定义", custom)]

    ok_all = True
    for i, (tag, q) in enumerate(cases):
        t0 = time.time()
        try:
            s = answer(graph, q, thread_id=f"test-{i}")
        except Exception as e:
            print(f"[{tag}] [FAIL] 异常: {type(e).__name__}: {e}")
            ok_all = False
            continue
        dt = time.time() - t0

        print(SEP)
        print(f"[{tag}]")
        print(f"  问题: {q}")
        print(f"  轨迹: {' -> '.join(s.get('trace', []))}")
        print(f"  路由: {s.get('route', '-')} | 检索轮次: {s.get('retrieve_rounds', 0)} "
              f"| 联网: {s.get('web_used', False)} | 幻觉校验: {s.get('hallucination_passed', '-')}")
        print(f"  回答: {(s.get('final_answer') or '')[:220]}")
        if s.get("citations"):
            print(f"  引用: {[(c['page'], c['source'][:20]) for c in s['citations']]}")
        print(f"  耗时: {dt:.1f}s")

    print(SEP)
    print("图行为验证完成")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
