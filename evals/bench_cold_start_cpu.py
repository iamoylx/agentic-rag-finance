"""
CPU 冷启动开销实测（T2，2026-10-20）——容器内 EMBED_DEVICE=cpu 的真实代价。

## 为什么需要这个脚本

Docker 镜像里不带模型权重与索引（走卷挂载），但**启动期仍然要加载模型、
并把语料编码成 dense 向量**。GPU 上 ~6s，CPU 上完全是另一个量级。
「容器能不能用」取决于这个数字，不能靠猜。

## 测什么

`python scripts/run_api.py` 到 `/health` 可用的真实成本，拆成可归因的几段：

| 段 | 说明 |
|---|---|
| [0] 导入期 | `import src.api.engine` 拉进 langgraph / langchain / chromadb 的一次性成本 |
| [1] 语料加载 | `corpus.jsonl` 命中缓存时直接还原，不重新解析 PDF |
| [2] BGE-M3 加载 | 权重进内存 |
| [3] Reranker 加载 | 第二个模型，同样吃内存 |
| [4] 全量编码 | `encode_dense` 对**全部** chunk 重算（改造前的瓶颈所在） |
| [5] BM25 构建 | 纯 Python，很快 |
| [6] Chroma 打开 | 已持久化，幂等覆盖 |

[4] 用**采样 + 外推**：先编码 N 条测出单条耗时，再乘全量规模。
外推会低估大 batch 的收益（batching 有摊薄），所以结果偏保守——
对一个「判断能不能接受」的决策来说，偏保守是正确方向。

## 一个自己的 bug，留作教训

本脚本第一版把 `from src.api.engine import Engine` 写在了计时段**里面**，
于是「语料加载」被标成 23.33s。实测 `build_corpus()` 只用 **0.48s**——
那 23 秒全是**导入**成本。归因错了会导致错误结论（「语料加载是瓶颈，要换存储格式」），
而真实瓶颈在别处。修法：导入移到计时之外，并单列「[0] 导入期」。

**给计量器本身写清楚的边界，比多测一个数字重要。**

## 语法

    EMBED_DEVICE=cpu python evals/bench_cold_start_cpu.py --sample 32

不消耗 LLM 额度、不发任何网络请求。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 必须在 import src.*（进而 import torch 相关模块）之前设好
os.environ.setdefault("EMBED_DEVICE", "cpu")


def _t() -> float:
    return time.perf_counter()


def main() -> int:
    ap = argparse.ArgumentParser(description="实测 CPU 冷启动各段开销")
    ap.add_argument("--sample", type=int, default=32,
                    help="编码采样条数（用于外推全量耗时）")
    a = ap.parse_args()

    # ---------- [0] 导入期（独立计量：它与「读语料」是两件事）----------
    t = _t()
    from src.api.engine import Engine
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.embeddings import BGEM3Embedder
    from src.retrieval.reranker import BGEReranker
    from src.retrieval.vector_store import ChromaVectorStore
    from src.core.config import settings
    t_import = _t() - t
    print(f"EMBED_DEVICE={settings.embed_device}  rerank_enabled={settings.rerank_enabled}")
    print(f"[0] 导入期（langgraph/langchain/chromadb/torch）：{t_import:.2f}s")
    print()

    # ---------- [1] 语料加载（缓存命中路径）----------
    eng = Engine()
    t = _t()
    docs, _ = eng.build_corpus()
    t_corpus = _t() - t
    n = len(docs)
    print(f"[1] 语料加载（corpus.jsonl 缓存命中）：{t_corpus:.2f}s，{n:,} 块")

    # ---------- [2] BGE-M3 加载 ----------
    t = _t()
    emb = BGEM3Embedder.get()
    t_embed_load = _t() - t
    print(f"[2] BGE-M3 加载：{t_embed_load:.2f}s")

    # ---------- [3] Reranker 加载 ----------
    t = _t()
    BGEReranker.get()
    t_rerank_load = _t() - t
    print(f"[3] Reranker 加载：{t_rerank_load:.2f}s")

    # ---------- [4] 全量编码（采样外推）----------
    sample = [d.page_content for d in docs[: a.sample]]
    t = _t()
    emb.encode_dense(sample)
    t_enc_sample = _t() - t
    per = t_enc_sample / max(len(sample), 1)
    est_full = per * n
    print(f"[4] 编码 {len(sample)} 条：{t_enc_sample:.2f}s → "
          f"{per * 1000:.1f} ms/块 → 外推全量 {n:,} 块 ≈ **{est_full:.0f}s "
          f"({est_full / 60:.1f} min)**")

    # ---------- [5] BM25 ----------
    t = _t()
    BM25Retriever().build(docs)
    t_bm25 = _t() - t
    print(f"[5] BM25 构建：{t_bm25:.2f}s")

    # ---------- [6] Chroma 现状（不重复 upsert，只看已有规模）----------
    t = _t()
    store = ChromaVectorStore()
    existing = store.count()
    t_chroma = _t() - t
    print(f"[6] Chroma 打开：{t_chroma:.2f}s，已有 {existing:,} 条向量")

    base = t_import + t_corpus + t_embed_load + t_rerank_load + t_bm25 + t_chroma
    print()
    print("-" * 60)
    print(f"**改造前**冷启动（[4] 为外推）≈ **{base + est_full:.0f}s "
          f"({(base + est_full) / 60:.1f} min)**")
    print(f"  其中 [4] 全量编码占 {est_full / (base + est_full) * 100:.0f}%")
    print(f"**改造后**（Chroma 已含全部向量，只补缺失块）≈ **{base:.1f}s**")
    print()
    if existing >= n:
        print(f"Chroma 里已有 {existing:,} 条 ≥ 语料 {n:,} 块 → 索引是持久化的，")
        print("改造后这次启动本可以一块都不编码。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
