"""
Mini RAG 原型：BM25 检索 + LLM 生成（带引用）。

用途：在 BGE-M3 混合检索与 LangGraph 图编排完成之前，
     先用「解析→分块→BM25→生成」跑通最小闭环，验证 LLM 链路与 Prompt 结构。
原理：Prompt 强制要求模型引用页码，为后面的「强制引用溯源 + 数值一致性校验」打地基。

运行：python scripts/mini_rag.py "你的问题"
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings            # noqa: E402
from src.core.llm import get_llm                # noqa: E402
from src.ingestion.chunker import FinancialChunker  # noqa: E402
from src.ingestion.pdf_parser import AnnualReportParser  # noqa: E402
from src.retrieval.bm25 import BM25Retriever    # noqa: E402

PROMPT_TMPL = """你是财报分析助手。仅根据下面提供的原文片段回答问题。

规则：
1. 只能使用片段中出现的信息，禁止编造数字。
2. 答案末尾必须标注引用来源，格式：[引用: 第X页]；片段不足以回答时，回复「原文中未找到相关信息」。

问题：{question}

原文片段：
{context}
"""


def build_context(docs, max_chars: int = 2400) -> str:
    """把检索到的块拼成 Prompt 上下文，标注每块页码。"""
    parts: list[str] = []
    used = 0
    for i, d in enumerate(docs, 1):
        text = d.page_content
        if used + len(text) > max_chars and parts:
            break
        parts.append(f"[片段{i} | 第{d.metadata.get('page')}页]\n{text}")
        used += len(text)
    return "\n\n".join(parts)


def main() -> int:
    settings.ensure_dirs()

    pdf_path = settings.raw_dir / "示例科技股份有限公司_2025年年度报告_测试样本.pdf"
    if not pdf_path.exists():
        print("测试 PDF 不存在，请先运行 scripts/smoke_test.py")
        return 1

    question = sys.argv[1] if len(sys.argv) > 1 else "公司2025年营业收入是多少？研发投入占营收比例呢？"

    blocks = AnnualReportParser(extract_tables=True).parse(pdf_path)
    docs = FinancialChunker().split(blocks, doc_meta={"company": "示例科技", "year": "2025"})
    retriever = BM25Retriever().build(docs)
    hits = retriever.search(question, top_k=4)

    context = build_context([d for d, _ in hits])
    prompt = PROMPT_TMPL.format(question=question, context=context)

    llm = get_llm(max_tokens=1024)
    answer = llm.invoke(prompt).content.strip()

    print("=" * 70)
    print(f"问题: {question}")
    print("-" * 70)
    print("检索命中的块:")
    for d, s in hits:
        print(f"  score={s:6.2f}  第{d.metadata.get('page')}页  ({d.metadata.get('block_type')})")
    print("-" * 70)
    print("回答:")
    print(answer)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
