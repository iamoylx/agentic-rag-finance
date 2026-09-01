"""
冒烟测试：端到端验证「PDF 解析 → 分块 → BM25 检索」链路。

用途：P1 地基自检。不依赖任何本地模型与 LLM，随时可跑。
原理：用结构化测试 PDF 驱动，逐段打印中间产物，任何一环异常都能当场看见。

运行：python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 把项目根目录加入 sys.path，使 `from src.xxx import` 可用
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings          # noqa: E402
from src.ingestion.chunker import FinancialChunker, summarize_chunks  # noqa: E402
from src.ingestion.pdf_parser import AnnualReportParser  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from scripts.make_test_pdf import build as build_test_pdf  # noqa: E402

SEP = "=" * 70


def main() -> int:
    settings.ensure_dirs()

    # ---------- Step 0: 准备测试 PDF ----------
    print(SEP)
    print("Step 0  准备测试载体")
    pdf_path = settings.raw_dir / "示例科技股份有限公司_2025年年度报告_测试样本.pdf"
    if not pdf_path.exists():
        build_test_pdf(pdf_path)
    print(f"  PDF: {pdf_path.name}  ({pdf_path.stat().st_size / 1024:.1f} KB)")

    # ---------- Step 1: 解析 ----------
    print(SEP)
    print("Step 1  PDF 解析（PyMuPDF 文本 + pdfplumber 表格）")
    t0 = time.time()
    parser = AnnualReportParser(extract_tables=True)
    blocks = parser.parse(pdf_path)
    dt = time.time() - t0

    n_text = sum(1 for b in blocks if b.block_type == "text")
    n_tab = sum(1 for b in blocks if b.block_type == "table")
    print(f"  耗时 {dt:.2f}s | 总块数 {len(blocks)} | 文本块 {n_text} | 表格块 {n_tab}")

    if n_tab == 0:
        print("  ⚠️  表格块为 0！表格链路未生效，检查 find_tables / extract_tables")
    # 展示一个表格块的 Markdown（验证表格→Markdown 转换正确）
    for b in blocks:
        if b.block_type == "table":
            print(f"\n  --- 表格块样例（第 {b.page} 页）---")
            for line in b.text.splitlines()[:5]:
                print(f"    {line}")
            break

    # 验证页眉页脚是否被清洗
    has_footer = any("— 1 —" in b.text or "— 2 —" in b.text for b in blocks)
    print(f"\n  页脚清洗: {'❌ 仍残留' if has_footer else '✅ 已清除'}")

    # ---------- Step 2: 分块 ----------
    print(SEP)
    print("Step 2  分块（递归切分 + 表格保护 + 上下文前缀注入）")
    chunker = FinancialChunker()
    docs = chunker.split(blocks, doc_meta={"company": "示例科技", "year": "2025"})
    st = summarize_chunks(docs)
    print(f"  总块数 {st.n_chunks} | 表格块 {st.n_table_chunks} | "
          f"平均长度 {st.avg_len} | 最大长度 {st.max_len}")

    # 验证上下文前缀确实注入了页码
    sample = next((d for d in docs if d.metadata.get("block_type") == "table"), None)
    if sample:
        print(f"\n  --- 表格 chunk 首行（验证上下文前缀）---")
        print(f"    {sample.page_content.splitlines()[0]}")
        print(f"    metadata.page = {sample.metadata.get('page')}")

    # ---------- Step 3: BM25 检索 ----------
    print(SEP)
    print("Step 3  BM25 检索（稀疏通路，不依赖模型）")
    t0 = time.time()
    retriever = BM25Retriever().build(docs)
    print(f"  索引构建耗时 {time.time() - t0:.3f}s | 文档数 {len(docs)}")

    queries = [
        "营业收入是多少",
        "归属于母公司股东的净利润",
        "研发投入占营业收入的比例",
        "应收账款账面价值",
        "毛利率",
    ]
    all_hit = True
    for q in queries:
        hits = retriever.search(q, top_k=3)
        print(f"\n  查询: {q}")
        if not hits:
            print("    ❌ 无命中")
            all_hit = False
            continue
        for i, (d, s) in enumerate(hits, 1):
            snippet = d.page_content.replace("\n", " ")[:62]
            print(f"    [{i}] score={s:.2f} p{d.metadata.get('page')} "
                  f"({d.metadata.get('block_type')}) {snippet}...")

    # ---------- 结论 ----------
    print(SEP)
    ok = n_tab > 0 and not has_footer and st.n_chunks > 0 and all_hit
    print(f"冒烟测试结果: {'✅ 全部通过' if ok else '❌ 存在失败项'}")
    print(SEP)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
