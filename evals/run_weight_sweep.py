"""
RRF 权重扫描（2026-09-17 补做，回答「权重怎么调的」这一面试追问）。

背景：此前 RRF 权重只有 2 个数据点（等权 → Recall@5 0.692；0.7:0.3 → 0.885），
      「0.7」是拍的而非扫出来的——面试官一追就露馅。本脚本把它补成一条真实曲线。

设计要点（评测脚本自身必须可自证）：
  阶段 0  基线复现自检：用项目自身的 4 条通路代码重跑，与 evals/retrieval_result.json
         记录的数字逐一对齐。对不上就说明本次环境/代码与当初不同，曲线不可信 → 直接退出。
  阶段 1  权重扫描：dense:bm25 从 1.0:0.0 扫到 0.0:1.0（11 点），每点测「不加精排 / 加精排」
         两套指标；并把端点与基线交叉验证（1.0:0.0 无精排应≈dense_only）。
  阶段 2  RRF k 值扫描：在「最优权重」与「仓库当前权重」两套配置下各扫一遍 k（平滑常数），
         看它对指标的敏感性——只在一套权重下验证 k 不敏感，不足以支撑部署配置的结论。

零 LLM 消耗：全部为本地嵌入 + BM25 + CrossEncoder。
运行：cd 项目根 && .venv/Scripts/python.exe evals/run_weight_sweep.py
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.engine import engine                      # noqa: E402
from src.core.config import settings                   # noqa: E402
from src.retrieval.hybrid import rrf_fuse              # noqa: E402

GOLD = ROOT / "evals" / "goldset.json"
BASELINE = ROOT / "evals" / "retrieval_result.json"
OUT = ROOT / "evals" / "weight_sweep_result.json"

_WS = re.compile(r"\s+")

# 扫描点：dense 权重从 1.0 递减到 0.0（bm25 = 1 - dense），步长 0.1
W_GRID = [round(1.0 - i * 0.1, 1) for i in range(11)]
K_GRID = [1, 3, 5, 10, 20, 40, 60, 100]

# 基线自检容差：指标按 26 条算，量化步长 1/26 ≈ 0.0385，允许 1 条的差异
TOL = 0.04


def _norm(s: str) -> str:
    """去全部空白：PDF 抽取常在数字/中文间插空格，匹配前必须归一（与 run_retrieval_eval 一致）。"""
    return _WS.sub("", s)


def relevant(chunks: list, item: dict) -> list[int]:
    """top-5 内每个位置是否命中（1/0）：块文本包含主口径答案子串或任一单位口径别名。"""
    pats = [item["answer_substr"]] + item.get("aliases", [])
    pats = [_norm(p) for p in pats if p]
    return [int(any(p in _norm(d.page_content) for p in pats)) for d in chunks]


def metrics(rel_vecs: list[list[int]]) -> dict:
    """与 run_retrieval_eval.py 完全同口径的三指标。"""
    n = len(rel_vecs)
    recall = sum(1 for v in rel_vecs if 1 in v) / n
    mrr = sum(1.0 / (v.index(1) + 1) for v in rel_vecs if 1 in v) / n
    dcg = sum(sum(rel / math.log2(pos + 2) for pos, rel in enumerate(v)) for v in rel_vecs) / n
    idcg = sum(sum(1.0 / math.log2(pos + 2) for pos in range(min(sum(v), len(v))))
               for v in rel_vecs) / n
    return {"recall5": recall, "mrr5": mrr, "ndcg5": dcg / idcg}


def _key(d) -> str:
    return f"{d.metadata.get('source')}|{d.metadata.get('page')}|{d.page_content}"


def main() -> None:
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    items = [it for it in gold["items"] if it["kind"] in ("numeric", "qualitative")]
    print(f"评测集: {len(items)} 条检索项\n")

    docs, _ = engine.build_corpus()
    print(f"语料: {len(docs)} 分块，建索引中…")
    t0 = time.time()
    engine._build(docs)
    r = engine.retriever
    print(f"索引就绪 ({time.time() - t0:.0f}s)\n")

    # ==================================================================
    # 阶段 0：基线复现自检（用项目自身代码，独立于下面的参数化路径）
    # ==================================================================
    print("=" * 78)
    print("阶段 0  基线复现自检（复跑 retrieval_result.json 的 4 条通路）")
    print("=" * 78)
    base_paths = {
        "dense_only":    lambda q: r.search_dense_only(q, top_k=5),
        "bm25_only":     lambda q: [d for d, _ in r.bm25.search(q, top_k=5)],
        "hybrid(RRF)":   lambda q: r.search_hybrid_no_rerank(q, top_k=5),
        "hybrid+rerank": lambda q: [d for d, _ in r.search(q, top_k=5)],
    }
    recorded = json.loads(BASELINE.read_text(encoding="utf-8"))["paths"]
    baseline = {}
    for name, fn in base_paths.items():
        rels = [relevant(fn(it["question"]), it) for it in items]
        m = metrics(rels)
        baseline[name] = m
        rec = recorded[name]
        ok = all(abs(m[k] - rec[k]) < TOL for k in ("recall5", "mrr5", "ndcg5"))
        flag = "✅ 对齐" if ok else "❌ 不一致"
        print(f"  {name:<15} R@5={m['recall5']:.3f} MRR@5={m['mrr5']:.3f} nDCG@5={m['ndcg5']:.3f}"
              f"   [记录 R@5={rec['recall5']:.3f}] {flag}")
        if not ok:
            print("\n⚠️ 基线与记录不符 → 环境/代码已变，扫描结果不可与 METRICS 混用，已退出。")
            sys.exit(1)
    print("  → 4 条通路全部对齐，本次环境可信，开始扫参\n")

    # ==================================================================
    # 预计算：双路 Top-20（与权重无关，只算一次）
    # ==================================================================
    print("预计算每题的 BM25 / dense Top-20 候选…")
    pre = []
    for it in items:
        q = it["question"]
        bm = [d for d, _ in r.bm25.search(q, top_k=settings.top_k_retrieve)]
        dn = [d for d, _ in r.store.search_dense(
            r.embedder.encode_dense([q])[0], top_k=settings.top_k_retrieve)]
        pre.append({"item": it, "q": q, "bm25": bm, "dense": dn})
    print("  完成\n")

    # 精排记忆化：候选集在不同权重间高度重叠，同一 (题号, 块) 只推理一次
    memo: dict[tuple[int, str], float] = {}

    def rerank_top(qi: int, q: str, cands: list, k_final: int) -> list:
        todo = [d for d in cands if (qi, _key(d)) not in memo]
        if todo:
            for d, s in r.reranker.rerank(q, todo):
                memo[(qi, _key(d))] = s
        return sorted(cands, key=lambda d: memo[(qi, _key(d))], reverse=True)[:k_final]

    def run(w_dense: float, w_bm25: float, use_rerank: bool,
            k: int | None = None) -> tuple[dict, list[str]]:
        kk = settings.rrf_k if k is None else k
        rels = []
        for qi, p in enumerate(pre):
            fused = rrf_fuse(p["bm25"], p["dense"], k=kk, weights=[w_bm25, w_dense])
            cands = [d for d, _ in fused[:settings.top_k_retrieve]]
            top = (rerank_top(qi, p["q"], cands, settings.top_k_final)
                   if use_rerank else cands[:settings.top_k_final])
            rels.append(relevant(top, p["item"]))
        miss = [items[j]["id"] for j, v in enumerate(rels) if 1 not in v]
        return metrics(rels), miss

    # ==================================================================
    # 阶段 1：权重扫描
    # ==================================================================
    print("=" * 78)
    print("阶段 1  dense : bm25 权重扫描（k=60 固定）")
    print("=" * 78)
    print(f"  {'dense':>6} {'bm25':>6} | {'R@5':>6} {'MRR@5':>6} {'nDCG@5':>7} (不加精排)"
          f" | {'R@5':>6} {'MRR@5':>6} {'nDCG@5':>7} (加精排) | 精排 miss")
    print("  " + "-" * 96)

    sweep = []
    t1 = time.time()
    for wd in W_GRID:
        wb = round(1.0 - wd, 1)
        m_no, _ = run(wd, wb, use_rerank=False)
        m_rr, miss_rr = run(wd, wb, use_rerank=True)
        sweep.append({"w_dense": wd, "w_bm25": wb, "no_rerank": m_no,
                      "rerank": m_rr, "rerank_misses": miss_rr})
        print(f"  {wd:>6.1f} {wb:>6.1f} | {m_no['recall5']:>6.3f} {m_no['mrr5']:>6.3f}"
              f" {m_no['ndcg5']:>7.3f}          | {m_rr['recall5']:>6.3f} {m_rr['mrr5']:>6.3f}"
              f" {m_rr['ndcg5']:>7.3f}       | {len(miss_rr)}")
    print(f"  （扫描耗时 {time.time() - t1:.0f}s，精排推理缓存 {len(memo)} 对）\n")

    # 端点交叉验证：1.0:0.0 无精排应≈dense_only；0.0:1.0 无精排应≈bm25_only
    e_dense = sweep[0]["no_rerank"]
    e_bm25 = sweep[-1]["no_rerank"]
    ok1 = abs(e_dense["recall5"] - baseline["dense_only"]["recall5"]) < TOL
    ok2 = abs(e_bm25["recall5"] - baseline["bm25_only"]["recall5"]) < TOL
    print("  交叉验证：")
    print(f"    1.0:0.0 无精排 R@5={e_dense['recall5']:.3f} vs dense_only "
          f"{baseline['dense_only']['recall5']:.3f}  {'✅' if ok1 else '❌'}")
    print(f"    0.0:1.0 无精排 R@5={e_bm25['recall5']:.3f} vs bm25_only "
          f"{baseline['bm25_only']['recall5']:.3f}  {'✅' if ok2 else '❌'}")

    # 选最优：精排 R@5 优先 → MRR@5 次之（业务含义：先保证答案在候选里，再让它靠前）
    def is_better(a: dict, b: dict) -> bool:
        return (a["recall5"], a["mrr5"]) > (b["recall5"], b["mrr5"])

    best = sweep[0]
    for s in sweep[1:]:
        if is_better(s["rerank"], best["rerank"]):
            best = s
    print(f"\n  ★ R@5 优先口径下的最优权重: dense:bm25 = {best['w_dense']}:{best['w_bm25']}"
          f"  (R@5={best['rerank']['recall5']:.3f} MRR@5={best['rerank']['mrr5']:.3f})")
    cur = next(s for s in sweep if abs(s["w_dense"] - settings.rrf_dense_weight) < 1e-9)
    print(f"  ● 当前仓库配置     : dense:bm25 = {cur['w_dense']}:{cur['w_bm25']}"
          f"  (R@5={cur['rerank']['recall5']:.3f} MRR@5={cur['rerank']['mrr5']:.3f})\n")

    # ==================================================================
    # 阶段 2：k 值扫描（在「最优权重」与「仓库当前权重」两套配置下各扫一遍）
    #   为什么必须两套都扫：k 只压平「两路名次差」的贡献，而两路候选集被选进
    #   候选窗口的构成本身随 dense:bm25 权重变化 → 只在一套权重下证明「k 不敏感」
    #   不足以支撑「部署配置也不用调 k」的结论（面试容易被追问这一点）。
    # ==================================================================
    print("=" * 78)
    print("阶段 2  RRF k 值扫描（最优权重 / 仓库当前权重，各扫一遍）")
    print("=" * 78)
    k_configs = [("best", best), ("current", cur)]
    k_sweep_by_cfg: dict[str, dict] = {}
    for tag, cfg in k_configs:
        label = "最优" if tag == "best" else "当前"
        print(f"\n  ── {label}配置 dense:bm25 = {cfg['w_dense']}:{cfg['w_bm25']} ──")
        print(f"  {'k':>5} | {'R@5':>6} {'MRR@5':>6} {'nDCG@5':>7} (加精排) | miss")
        print("  " + "-" * 62)
        rows = []
        for k in K_GRID:
            m_rr, miss_rr = run(cfg["w_dense"], cfg["w_bm25"], use_rerank=True, k=k)
            rows.append({"k": k, "rerank": m_rr, "misses": miss_rr})
            print(f"  {k:>5} | {m_rr['recall5']:>6.3f} {m_rr['mrr5']:>6.3f}"
                  f" {m_rr['ndcg5']:>7.3f}       | {len(miss_rr)}")
        k_sweep_by_cfg[tag] = {"w_dense": cfg["w_dense"], "w_bm25": cfg["w_bm25"],
                               "rows": rows}
    k_sweep = k_sweep_by_cfg["best"]["rows"]   # 兼容旧字段
    print()

    # ==================================================================
    # 阶段 3：候选集稳定性——解释「k 值为什么扫不出差异」
    # ==================================================================
    print("=" * 78)
    print("阶段 3  候选集稳定性分析（为什么 k 值不影响最终指标）")
    print("=" * 78)
    ref_k = settings.rrf_k
    cand_sets_by_cfg: dict[str, dict[int, list[frozenset]]] = {}
    for tag, cfg in k_configs:
        per_k = {}
        for k in K_GRID:
            sets = []
            for p in pre:
                fused = rrf_fuse(p["bm25"], p["dense"], k=k,
                                 weights=[cfg["w_bm25"], cfg["w_dense"]])
                sets.append(frozenset(_key(d) for d, _ in fused[:settings.top_k_retrieve]))
            per_k[k] = sets
        cand_sets_by_cfg[tag] = per_k

    uni, inter = [], []
    for p in pre:
        b = {_key(d) for d in p["bm25"]}
        d_ = {_key(d) for d in p["dense"]}
        uni.append(len(b | d_))
        inter.append(len(b & d_))
    avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(f"  两路各取 Top-{settings.top_k_retrieve}：平均并集 {avg(uni):.1f} 个块，"
          f"平均重叠 {avg(inter):.1f} 个块")
    print(f"  （并集 {avg(uni):.1f} > 候选窗口 {settings.top_k_retrieve} → "
          f"融合后只淘汰 {avg(uni) - settings.top_k_retrieve:.1f} 个两路都排最末的块，与 k 无关）")

    k_stability_by_cfg: dict[str, list] = {}
    for tag, cfg in k_configs:
        label = "最优" if tag == "best" else "当前"
        ref_sets = cand_sets_by_cfg[tag][ref_k]
        rows = k_sweep_by_cfg[tag]["rows"]
        detail = []
        print(f"\n  ── {label}配置 dense:bm25 = {cfg['w_dense']}:{cfg['w_bm25']} ──")
        print(f"  {'k':>5} | 候选 Top-20 集合与 k={ref_k} 完全一致的题数 | 指标是否变化")
        print("  " + "-" * 66)
        for k in K_GRID:
            same = sum(1 for a, b in zip(cand_sets_by_cfg[tag][k], ref_sets) if a == b)
            m = next(x["rerank"] for x in rows if x["k"] == k)
            unchanged = abs(m["recall5"] - cfg["rerank"]["recall5"]) < 1e-9
            detail.append({"k": k, "identical_candidate_sets": same, "n": len(pre),
                           "metrics_unchanged": unchanged})
            print(f"  {k:>5} | {same}/{len(pre):>3} 题完全相同"
                  f"{'':<22} | {'否（指标一致）' if unchanged else '是 ⚠️'}")
        k_stability_by_cfg[tag] = detail
    k_stability = k_stability_by_cfg["best"]   # 兼容旧字段
    print()

    # 安全带区间（精排模式 R@5 达到最优的权重区间）与断崖
    top_r = max(s["rerank"]["recall5"] for s in sweep)
    band = [s["w_dense"] for s in sweep if s["rerank"]["recall5"] >= top_r - 1e-9]
    cliff = None
    for a, b in zip(sweep, sweep[1:]):
        if b["rerank"]["recall5"] < a["rerank"]["recall5"] - 0.1:
            cliff = {"from": a["w_dense"], "to": b["w_dense"],
                     "r5_from": a["rerank"]["recall5"], "r5_to": b["rerank"]["recall5"]}
            break
    print(f"  ★ 安全带：精排模式下 R@5 达到最优({top_r:.3f}) 的 dense 权重区间 = "
          f"[{min(band):.1f}, {max(band):.1f}]（{len(band)} 档）")
    if cliff:
        print(f"  ★ 断崖：dense {cliff['from']:.1f} → {cliff['to']:.1f} 时 R@5 "
              f"{cliff['r5_from']:.3f} → {cliff['r5_to']:.3f}"
              f"（候选集被 BM25 噪声污染，精排无法找回未进候选的答案）\n")

    # ==================================================================
    # 阶段 4：0.5 vs 0.7 的配对显著性——「差 0.040 是改进还是噪声？」
    # ==================================================================
    print("=" * 78)
    print("阶段 4  配对检验：dense 0.5 与 0.7 的 MRR 差异是否显著（n=26 很小，必须验）")
    print("=" * 78)

    def rr_vec(w_dense: float, w_bm25: float) -> list[float]:
        """逐题的 reciprocal rank（未命中记 0），与 metrics() 的 MRR 同口径。"""
        out = []
        for qi, p in enumerate(pre):
            fused = rrf_fuse(p["bm25"], p["dense"], k=settings.rrf_k,
                             weights=[w_bm25, w_dense])
            cands = [d for d, _ in fused[:settings.top_k_retrieve]]
            top = rerank_top(qi, p["q"], cands, settings.top_k_final)
            v = relevant(top, p["item"])
            out.append(1.0 / (v.index(1) + 1) if 1 in v else 0.0)
        return out

    r5, r7 = rr_vec(0.5, 0.5), rr_vec(0.7, 0.3)
    diff = [x - y for x, y in zip(r5, r7)]
    changed = [(items[i]["id"], y, x) for i, (x, y) in enumerate(zip(r5, r7))
               if abs(x - y) > 1e-9]
    print(f"  逐题比对：命中位次发生变化的题数 = {len(changed)}/{len(items)}")
    for cid, old, new in changed:
        print(f"    {cid:<26} 0.7→rr={old:.3f}   0.5→rr={new:.3f}")

    n = len(diff)
    mean_d = sum(diff) / n
    random.seed(0)
    boot = sorted(sum(diff[random.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    lo, hi = boot[int(0.025 * 2000)], boot[int(0.975 * 2000)]
    crosses = lo <= 0 <= hi
    print(f"\n  平均 ΔMRR（0.5 − 0.7）= {mean_d:+.4f}")
    print(f"  配对 bootstrap 95% CI = [{lo:+.4f}, {hi:+.4f}]（2000 次重采样，seed=0）")
    print(f"  → CI {'跨越 0 ⇒ 差异不显著，不能据此认为 0.5 更优' if crosses else '不跨 0 ⇒ 差异显著'}")
    print(f"  等价解读：ΔMRR×n = {mean_d * n:+.2f}，即约 {abs(mean_d) * n / 0.5:.1f} 道题"
          f"从第 1 名掉到第 2 名就能造成这个量级的差异\n")

    OUT.write_text(json.dumps({
        "n": len(items),
        "note": "基线复现自检通过后才产出；dense_only/bm25_only 为端点锚点",
        "baseline_reproduced": baseline,
        "weight_grid": sweep,
        "best_by_recall": {"w_dense": best["w_dense"], "w_bm25": best["w_bm25"],
                           "rerank": best["rerank"]},
        "current_config": {"w_dense": cur["w_dense"], "w_bm25": cur["w_bm25"],
                           "rerank": cur["rerank"]},
        "safe_band": {"w_dense_min": min(band), "w_dense_max": max(band),
                      "best_r5": top_r},
        "cliff": cliff,
        "k_grid": k_sweep,
        "k_grid_by_config": {
            t: {"w_dense": v["w_dense"], "w_bm25": v["w_bm25"], "rows": v["rows"]}
            for t, v in k_sweep_by_cfg.items()},
        "k_stability": {"avg_union": avg(uni), "avg_overlap": avg(inter),
                        "detail": k_stability},
        "k_stability_by_config": k_stability_by_cfg,
        "paired_test": {"compare": "dense 0.5 vs 0.7（精排模式）",
                        "changed_questions": len(changed), "n": n,
                        "mean_delta_mrr": mean_d, "ci95": [lo, hi],
                        "significant": not crosses,
                        "detail": [{"id": c, "rr_07": o, "rr_05": x} for c, o, x in changed]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {OUT.relative_to(ROOT)}")
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
