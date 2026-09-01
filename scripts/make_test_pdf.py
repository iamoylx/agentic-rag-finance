"""
生成模拟年报 PDF（测试载体）。

用途：真实年报还没下载时，先用结构化测试 PDF 打通「解析→分块→检索」全链路。
原理：表格必须画出真实线条，因为 pdfplumber 的 extract_tables() 依赖线条或文本对齐来识别表格；
      纯空格对齐的文本表格检测不到，会漏测表格链路。
"""
from __future__ import annotations

from pathlib import Path

import fitz

# 简体中文字体（PyMuPDF 内置，无需额外字体文件）
CN_FONT = "china-s"

COMPANY = "示例科技股份有限公司"
YEAR = 2025


def _draw_table(
    page: fitz.Page,
    x0: float,
    y0: float,
    rows: list[list[str]],
    col_widths: list[float],
    row_height: float = 22.0,
    font_size: float = 8.0,
) -> float:
    """画一个带线条的表格并返回占用的高度。

    用途：生成可被 pdfplumber 识别的真表格。
    原理：先画横线再画竖线形成网格，然后在每个单元格内居中插入文本。
    """
    n_rows, n_cols = len(rows), len(col_widths)
    total_w = sum(col_widths)
    total_h = row_height * n_rows

    # 横线
    for i in range(n_rows + 1):
        y = y0 + i * row_height
        page.draw_line(fitz.Point(x0, y), fitz.Point(x0 + total_w, y), width=0.6)
    # 竖线
    x = x0
    for w in col_widths:
        page.draw_line(fitz.Point(x, y0), fitz.Point(x, y0 + total_h), width=0.6)
        x += w
    page.draw_line(fitz.Point(x0 + total_w, y0), fitz.Point(x0 + total_w, y0 + total_h), width=0.6)

    # 填字
    for r, row in enumerate(rows):
        cx = x0
        for c, cell in enumerate(row):
            if c >= n_cols:
                break
            tx = cx + 4
            ty = y0 + r * row_height + row_height * 0.7
            page.insert_text(
                fitz.Point(tx, ty), str(cell),
                fontname=CN_FONT, fontsize=font_size,
            )
            cx += col_widths[c]
    return total_h


def _draw_para(page: fitz.Page, x0: float, y0: float, text: str,
               font_size: float = 10.5, leading: float = 16.0) -> float:
    """插入一段正文，返回占用高度。"""
    ty = y0
    for line in text.splitlines():
        page.insert_text(fitz.Point(x0, ty), line, fontname=CN_FONT, fontsize=font_size)
        ty += leading
    return ty - y0


def build(out_path: Path) -> None:
    doc = fitz.open()
    W, H = 595, 842  # A4

    # ---------- 第 1 页：公司概况 + 主要财务指标表 ----------
    p = doc.new_page(width=W, height=H)
    p.insert_text(fitz.Point(180, 60), f"{COMPANY}", fontname=CN_FONT, fontsize=16)
    p.insert_text(fitz.Point(200, 82), f"{YEAR} 年年度报告", fontname=CN_FONT, fontsize=13)
    y = _draw_para(p, 60, 130,
                   "第一节  释义及重大风险提示\n\n"
                   "本公司董事会、监事会及董事、监事、高级管理人员保证年度报告内容的真实、准确、完整，"
                   "不存在虚假记载、误导性陈述或重大遗漏，并承担个别和连带的法律责任。\n"
                   "本报告中所涉及的未来计划、发展战略等前瞻性陈述，不构成公司对投资者的实质承诺。")
    y += 140

    p.insert_text(fitz.Point(60, y + 10),
                  "第二节  主要会计数据和财务指标", fontname=CN_FONT, fontsize=12)
    rows = [
        ["项目", f"{YEAR}年", f"{YEAR - 1}年", "同比增减"],
        ["营业收入（元）", "12,345,678,901.23", "10,987,654,321.00", "12.36%"],
        ["归属于母公司股东的净利润（元）", "1,234,567,890.12", "1,098,765,432.10", "12.36%"],
        ["扣除非经常性损益后的净利润（元）", "1,100,000,000.00", "980,000,000.00", "12.24%"],
        ["经营活动产生的现金流量净额（元）", "2,345,678,901.23", "2,100,000,000.00", "11.70%"],
        ["总资产（元）", "45,678,901,234.56", "40,123,456,789.01", "13.85%"],
        ["加权平均净资产收益率", "18.42%", "17.15%", "增加1.27个百分点"],
        ["基本每股收益（元/股）", "3.86", "3.44", "12.21%"],
    ]
    _draw_table(p, 50, y + 30, rows, [158, 145, 145, 75])

    # 页脚（用于验证页眉页脚清洗逻辑）
    p.insert_text(fitz.Point(250, H - 40), "— 1 —", fontname=CN_FONT, fontsize=9)

    # ---------- 第 2 页：主营业务分行业构成（跨页表格的简化版） ----------
    p2 = doc.new_page(width=W, height=H)
    p2.insert_text(fitz.Point(150, 60), "第三节  主营业务分析",
                   fontname=CN_FONT, fontsize=14)
    y2 = _draw_para(p2, 60, 100,
                    "报告期内，公司实现营业收入 12,345,678,901.23 元，同比增长 12.36%；\n"
                    "实现归属于母公司股东的净利润 1,234,567,890.12 元，同比增长 12.36%。\n"
                    "毛利率为 42.18%，较上年同期提升 1.05 个百分点。")
    rows2 = [
        ["分行业", "营业收入（元）", "营业成本（元）", "毛利率", "同比增减"],
        ["智能制造装备", "8,000,000,000.00", "4,500,000,000.00", "43.75%", "10.20%"],
        ["工业软件", "2,800,000,000.00", "1,200,000,000.00", "57.14%", "15.60%"],
        ["技术服务", "1,545,678,901.23", "943,000,000.00", "38.99%", "8.40%"],
        ["合计", "12,345,678,901.23", "6,643,000,000.00", "46.19%", "12.36%"],
    ]
    _draw_table(p2, 60, y2 + 130, rows2, [110, 120, 120, 80, 80])
    p2.insert_text(fitz.Point(250, H - 40), "— 2 —", fontname=CN_FONT, fontsize=9)

    # ---------- 第 3 页：研发投入 + 风险（纯文本，检验文本分块） ----------
    p3 = doc.new_page(width=W, height=H)
    p3.insert_text(fitz.Point(150, 60), "第四节  研发投入与风险因素",
                   fontname=CN_FONT, fontsize=14)
    _draw_para(p3, 60, 100,
               "报告期内，公司研发投入合计 1,876,543,210.98 元，占营业收入的比例为 15.20%，\n"
               "较上年同期提升 0.83 个百分点。研发人员数量 4,286 人，占公司总人数的 32.15%。\n\n"
               "公司面临的主要风险包括：\n"
               "1. 宏观经济波动导致下游客户需求放缓的风险；\n"
               "2. 关键原材料价格上涨导致毛利率下降的风险；\n"
               "3. 核心技术迭代不及预期导致产品竞争力下降的风险；\n"
               "4. 应收账款余额较大导致的信用减值风险。报告期末应收账款账面价值\n"
               "   为 5,678,901,234.56 元，占总资产比例为 12.43%。\n\n"
               "针对上述风险，公司已制定应对措施：加强客户信用管理、推进核心零部件国产化替代、\n"
               "持续加大研发投入以保持技术领先性。")
    p3.insert_text(fitz.Point(250, H - 40), "— 3 —", fontname=CN_FONT, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    doc.close()


if __name__ == "__main__":
    from src.core.config import settings

    out = settings.raw_dir / f"{COMPANY}_{YEAR}年年度报告_测试样本.pdf"
    build(out)
    print(f"已生成测试 PDF: {out}")
    print(f"页数: 3 | 大小: {out.stat().st_size / 1024:.1f} KB")
