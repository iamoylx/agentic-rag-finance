"""
引用准确率评测（P1-S8 复盘扩展）：从已落盘的端到端评测结果里解析答案引用，对照语料出处。

语法：答案文本末尾带 [引用: 第X页、第Y页]，与金标出处对照（问题文本对齐，
     兼容早期金标 id 撞车的历史结果）。
用途：RAG 系统的「可溯源」指标——答案对了但引用是幻觉页，用户依然没法核数。

双口径设计（指标本身的洞察，面试可讲）：
  ① 引用准确率（严格）：金标「主要会计数据页」被引用。同一数值常出现在年报多个章节
     （主要会计数据页 / MD&A / 附注），模型引用同值的其他页不算错但也不算命中——
     衡量「溯源到指定权威出处」的能力。
  ② 引用有效率（防幻觉）：至少一个被引用页在语料里真实包含该数值。
     语料扫描构建「有效页集合」（与金标自检同一套去空白 + 别名匹配）。
     引用错页 = 模型编造出处 = 严格 fail；引用同值页 = 语义有效。
  两者差值 ≈ 页码多义性（同一数值散布多页）的贡献。

诚实说明：本脚本对 evals/e2e_result.json（S7 真实 LLM 跑出的 26 条）做离线解析，
     不新消耗 LLM 配额；端到端结果更新后需重跑本脚本。
运行：env -u PYTHONPATH python evals/run_citation_eval.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
_CITE = re.compile(r"引用[:：][^\]]*\]")
_PAGE = re.compile(r"第(\d+)页")
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub("", s)


def main() -> None:
    e2e = json.loads((ROOT / "evals" / "e2e_result.json").read_text(encoding="utf-8"))
    gold = json.loads((ROOT / "evals" / "goldset.json").read_text(encoding="utf-8"))
    by_q = {it["question"]: it for it in gold["items"]}   # 历史结果 id 有重复，按问题对齐

    # 扫语料建「有效页集合」：页文本包含主口径答案或别名 → 该页是有效出处（本地算力，零 LLM）
    from src.api.engine import engine
    docs, _ = engine.build_corpus()
    pages: dict[tuple[str, int], str] = {}    # (source_pdf, page) -> 拼接文本
    for d in docs:
        meta = d.metadata or {}
        key = (meta.get("source", ""), int(meta.get("page", -1)))
        pages[key] = pages.get(key, "") + d.page_content
    for k in pages:
        pages[k] = _norm(pages[k])

    def valid_pages(it: dict) -> set[int]:
        pats = [_norm(p) for p in [it["answer_substr"], *it.get("aliases", [])] if p]
        return {pg for (src, pg), txt in pages.items()
                if src == it["source_pdf"] and any(p in txt for p in pats)}

    rows = []
    strict_hit = valid_hit = 0
    for r in e2e["results"]:
        it = by_q[r["q"]]
        cites = [int(p) for p in
                 _PAGE.findall("".join(_CITE.findall(r["answer"])))]
        vp = valid_pages(it)
        strict = bool(cites) and it["source_page"] in cites
        valid = bool(cites) and bool(set(cites) & vp)
        strict_hit += strict
        valid_hit += valid
        rows.append({"id": it["id"], "gold_page": it["source_page"],
                     "valid_pages": sorted(vp), "cited": cites,
                     "strict_ok": strict, "valid_ok": valid})

    n = len(rows)
    print(f"===== 引用双口径评测（{n} 条）=====")
    print(f"  引用准确率（严格命中金标页）: {strict_hit}/{n} = {strict_hit/n:.3f}")
    print(f"  引用有效率（引用页确含该数值）: {valid_hit}/{n} = {valid_hit/n:.3f}")
    for x in rows:
        if not x["strict_ok"]:
            tag = "无引用" if not x["cited"] else (
                "幻觉引用" if not x["valid_ok"] else "同值页")
            print(f"  MISS[{tag}] {x['id']}  gold={x['gold_page']}  cited={x['cited']}")
    (ROOT / "evals" / "citation_result.json").write_text(
        json.dumps({"n": n, "strict": {"hit": strict_hit, "acc": strict_hit / n},
                    "valid": {"hit": valid_hit, "acc": valid_hit / n},
                    "results": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入 evals/citation_result.json")


if __name__ == "__main__":
    main()
