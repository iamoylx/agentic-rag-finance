"""
评测集构建（P1-S7）。

语法：数值金标从年报「主要会计数据」表格人工核对后硬编码（来源已逐行验证）；
     定性金标的出处页用关键词检索自动定位。
用途：产出 evals/goldset.json——30 条 = 26 条检索评测（问题+答案子串+出处页）+ 4 条诱导幻觉陷阱。
     数字全部来自真实年报，禁止估算。
运行：env -u PYTHONPATH python evals/build_goldset.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "evals" / "goldset.json"

PDF_MTT = "贵州茅台_2024年年度报告.pdf"
PDF_CATL = "宁德时代_2024年年度报告.pdf"
PDF_BYD = "比亚迪_2024年年度报告.pdf"


def _q(company: str, question: str, answer: str, pdf: str, page: int, kind: str = "numeric",
       seq: int = 0) -> dict:
    # 单位口径别名：同一数值在年报不同报表里可能是「元」或「千元」口径
    # （如主要会计数据页 777,102,455,000.00 元 ↔ 合并利润表 777,102,455 千元），
    # 检索命中任一口径都算命中；主口径仍须通过「在指定页存在」的自检
    aliases = []
    if kind == "numeric" and answer.endswith(",000.00"):
        aliases.append(answer[:-7])           # 去掉千位段的 ,000.00 → 千元口径
    if kind == "numeric" and re.fullmatch(r"-?\d+\.\d+", answer):
        aliases.append(answer + "%")
    # id 带序号：同一页常有多道题，裸 {company}-{kind}-{page} 会撞 id（评测卫生 bug，S7 复盘修复）
    suf = f"-{seq}" if seq else ""
    return {"id": f"{company}-{kind}-{page}{suf}", "company": company, "question": question,
            "answer_substr": answer, "aliases": aliases,
            "source_pdf": pdf, "source_page": page, "kind": kind}


def build() -> dict:
    items: list[dict] = []
    m, c, b = PDF_MTT, PDF_CATL, PDF_BYD

    # ---------- 贵州茅台 p5（单位：元）----------
    items += [
        _q("茅台", "贵州茅台2024年营业收入是多少？", "170,899,152,276.34", m, 5),
        _q("茅台", "贵州茅台2024年归属于上市公司股东的净利润是多少？", "86,228,146,421.62", m, 5),
        _q("茅台", "贵州茅台2024年经营活动产生的现金流量净额是多少？", "92,463,692,168.43", m, 5),
        _q("茅台", "贵州茅台2024年末的总资产是多少？", "298,944,579,918.70", m, 5),
        _q("茅台", "贵州茅台2024年基本每股收益是多少？", "68.64", m, 5),
        _q("茅台", "贵州茅台2024年营业收入同比增长了多少？", "15.71", m, 5),
        _q("茅台", "贵州茅台2024年归母净利润的同比增速是多少？", "15.38", m, 5),
    ]
    # ---------- 宁德时代 p9（单位：千元）----------
    items += [
        _q("宁德", "宁德时代2024年营业收入（单位：千元）是多少？", "362,012,554", c, 9),
        _q("宁德", "宁德时代2024年归属于上市公司股东的净利润（单位：千元）是多少？", "50,744,682", c, 9),
        _q("宁德", "宁德时代2024年经营活动产生的现金流量净额（单位：千元）是多少？", "96,990,345", c, 9),
        _q("宁德", "宁德时代2024年基本每股收益是多少？", "11.58", c, 9),
        _q("宁德", "宁德时代2024年营业收入同比变化率是多少？", "9.70", c, 9),
        _q("宁德", "宁德时代2024年净利润的同比增速是多少？", "15.01", c, 9),
        _q("宁德", "宁德时代2024年第四季度的营业收入（单位：千元）是多少？", "102,967,805", c, 9),
    ]
    # ---------- 比亚迪 p10/p11（单位：元）----------
    items += [
        _q("比亚迪", "比亚迪2024年营业收入是多少？", "777,102,455,000.00", b, 10),
        _q("比亚迪", "比亚迪2024年归属于上市公司股东的净利润是多少？", "40,254,346,000.00", b, 11),
        _q("比亚迪", "比亚迪2024年经营活动产生的现金流量净额是多少？", "133,453,873,000.00", b, 11),
        _q("比亚迪", "比亚迪2024年基本每股收益是多少？", "13.84", b, 11),
        _q("比亚迪", "比亚迪2024年扣除非经常性损益后的净利润是多少？", "36,982,887,000.00", b, 11),
        _q("比亚迪", "比亚迪2024年净利润的同比增速是多少？", "34.00", b, 11),
        _q("比亚迪", "比亚迪2024年营业收入同比增速是多少？", "29.02", b, 10),
        _q("比亚迪", "比亚迪2024年经营现金流同比变化率是多少？", "21.37", b, 11),
        _q("比亚迪", "比亚迪2024年年报的审计会计师事务所是哪家？", "安永华明", b, 10, "qualitative"),
    ]
    # ---------- 定性问题（出处页自动定位：正文页里同时含关键词与业务词）----------
    qual = [
        ("贵州茅台的主要产品分为哪两大类？", "茅台酒及系列酒", m, ["茅台酒"]),
        ("宁德时代的主营业务是什么？", "动力电池", c, ["主要业务"]),
        ("比亚迪的业务主要覆盖哪些领域？", "新能源汽车", b, ["主要业务"]),
    ]
    for question, kw, pdf, biz in qual:
        page = _locate(pdf, kw, biz)
        it = _q(question.split("的")[0], question, kw, pdf, page, "qualitative")
        if kw == "茅台酒及系列酒":
            # 同义表述别名：语义相同的不同措辞（"和/与/、"）不应判错
            it["aliases"] += ["茅台酒和系列酒", "茅台酒与系列酒", "茅台酒、系列酒"]
        items.append(it)

    # ---------- 诱导幻觉陷阱（原文不存在，期望系统拒绝回答）----------
    traps = [
        _q("茅台", "贵州茅台2024年员工平均薪酬是多少？", "", m, -1, "trap"),
        _q("宁德", "宁德时代2024年末的市值是多少？", "", c, -1, "trap"),
        _q("宁德", "宁德时代2024年光伏组件的销量是多少？", "", c, -1, "trap"),
        _q("比亚迪", "比亚迪2024年股价最高是多少？", "", b, -1, "trap"),
    ]

    # ---------- 自检：数值金标必须真的在指定页原文里出现 ----------
    bad = []
    for it in items:
        d = fitz.open(RAW / it["source_pdf"])
        text = d[it["source_page"] - 1].get_text().replace(" ", "").replace("\n", "")
        d.close()
        if it["answer_substr"].replace(" ", "") not in text:
            bad.append(it["id"])
    if bad:
        raise SystemExit(f"金标自检失败（答案子串不在指定页）: {bad}")

    return {"items": items, "traps": traps}


def _locate(pdf: str, kw: str, biz: list[str], start: int = 5) -> int:
    """定位定性问题的出处页：正文页（≥start）里关键词与业务词同页；找不到则退化为关键词首页。"""
    d = fitz.open(RAW / pdf)
    fallback = -1
    try:
        for i in range(start - 1, d.page_count):
            t = d[i].get_text()
            if kw in t:
                if fallback < 0:
                    fallback = i + 1
                if any(x in t for x in biz):
                    return i + 1
    finally:
        d.close()
    if fallback < 0:
        raise SystemExit(f"{pdf} 中未找到关键词 {kw}")
    return fallback


if __name__ == "__main__":
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {OUT}: {len(data['items'])} 条检索评测 + {len(data['traps'])} 条陷阱，金标自检通过")
