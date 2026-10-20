"""检索层的两条契约（T2，2026-10-20）：向量跳过逻辑 + `search()` 返回同型。

## 为什么需要它

CPU 部署下冷启动的最贵一项是**重算全部 chunk 的 dense 向量**：实测 10014 块
≈ 19 分钟（`evals/bench_cold_start_cpu.py`）。而 Chroma 是持久化的，重启后
这些向量本来就在库里。

所以 `HybridRetriever` 改成：先问 `ChromaVectorStore.present_ids(docs)`「哪些块已入库」，
只对**缺失的**块编码。这个优化把 19 分钟压到几秒。

**但整条优化建立在一个前提上**：`present_ids()` 必须如实报告。
如果它少报（把已有的报成缺失）→ 变慢，能忍；
如果它**多报**（把缺失的报成已有）→ **静默漏编码，检索结果悄悄退化**，
而且不会报错、不会变慢，只是答得变差。这种错只能靠测试钉住。

这正是本项目反复出现的同一类风险：**「检查通过了」与「检查真的发生了」不可区分**。
所以这里不只测「返回非空」，而是**逐 id 断言**：库里有的必须在结果里、
库里没有的必须不在结果里。

### ② `search()` 两条分支必须返回同一类型

`rerank_enabled=False`（CPU 部署选项）新增后立刻暴露了一个既存不一致：
开精排返回 `[(Document, score)]`，关精排却返回**裸 Document**。
调用方 `src/graph/nodes.py` 是 `docs = [d for d, _ in hits]` → 当场
`ValueError: too many values to unpack`。

这个分支此前实际不可达，所以从来没人发现。
**新开关的价值之一，就是把不可达分支变成可达，然后暴露它的谎言。**

零外部依赖（临时目录 + 假 embedder/reranker，不加载 4.3 GB 真模型）。
"""
import sys
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.vector_store import ChromaVectorStore, doc_id  # noqa: E402


def _doc(text: str, page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": "t.pdf", "page": page})


def _store(tmp_path, name: str) -> ChromaVectorStore:
    # 每个用例独立 collection + 独立持久化目录，避免互相污染
    return ChromaVectorStore(persist_dir=tmp_path, collection_name=name)


def test_present_ids_empty_store_returns_nothing(tmp_path):
    """空库必须一条都不报——否则首跑就会跳过全部编码，建成一个空索引。"""
    store = _store(tmp_path, "empty")
    docs = [_doc("a"), _doc("b")]
    assert store.present_ids(docs) == set()


def test_present_ids_reports_exactly_what_is_stored(tmp_path):
    """逐 id 断言：入库的必须在，没入库的必须不在。

    只断言「集合非空」是没用的——多报一个就是静默漏编码。
    """
    store = _store(tmp_path, "partial")
    stored = [_doc("甲"), _doc("乙")]
    store.upsert(stored, [[0.1] * 8, [0.2] * 8])

    absent = _doc("丙")
    have = store.present_ids(stored + [absent])

    assert have == {doc_id(stored[0]), doc_id(stored[1])}
    assert doc_id(absent) not in have, "没入库的块被报成已存在 → 会静默漏编码"


def test_doc_id_is_stable_and_content_sensitive(tmp_path):
    """id 必须稳定（同内容同 id）且对内容/页码敏感（否则改稿不重建）。"""
    d = _doc("同样的一段话", page=3)
    assert doc_id(d) == doc_id(Document(page_content="同样的一段话",
                                        metadata={"source": "t.pdf", "page": 3}))

    assert doc_id(d) != doc_id(_doc("同样的一段话", page=4)), "页码变了 id 应变"
    assert doc_id(d) != doc_id(_doc("同样的一段话（改）", page=3)), "正文变了 id 应变"


def test_duplicate_content_shares_one_id(tmp_path):
    """同页同内容的重复块共用一个 id —— upsert 靠它幂等，不能翻倍。

    真实年报里「同页同内容」确实存在（重复段落 / 相同表格），
    这也是 Chroma 里向量数(9537) 少于语料块数(10014) 的原因。
    """
    a, b = _doc("重复段落"), _doc("重复段落")
    assert doc_id(a) == doc_id(b)

    store = _store(tmp_path, "dup")
    store.upsert([a, b], [[0.1] * 8, [0.2] * 8])
    assert store.count() == 1


# ---------------- 跳过逻辑：首跑全编码、二跑零编码 ----------------
class _FakeEmbedder:
    """只数「被要求编码了多少条」，不做真计算——真模型在 CPU 上要 19 分钟。"""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def encode_dense(self, texts):
        self.calls.append(len(texts))
        return [[0.1] * 8 for _ in texts]

    @property
    def total_encoded(self) -> int:
        return sum(self.calls)


def _build(tmp_path, docs, embedder):
    """在临时 chroma 目录上构建检索器（不碰真实 data/chroma，不加载真模型）。"""
    from src.retrieval.hybrid import HybridRetriever

    # use_rerank=False：精排模型与本次验证无关，省掉一次权重加载
    return HybridRetriever(
        docs,
        use_rerank=False,
        embedder=embedder,
        store=_store(tmp_path, "build"),
    )


def test_first_build_encodes_everything_second_build_encodes_nothing(tmp_path):
    """这条是 CPU 冷启动优化（19 分钟 → 11 秒）的核心契约。

    两个方向都要断言，缺一个就不成立：
      · 首跑必须**全量编码**，否则索引是空的 → 检索静默退化
      · 二跑必须**零编码**，否则优化白做 → 容器冷启动仍然不可用
    """
    docs = [_doc(f"块{i}", page=i) for i in range(1, 6)]

    first = _FakeEmbedder()
    r1 = _build(tmp_path, docs, first)
    assert first.total_encoded == len(docs), "首跑没有全量编码，索引会是空的"
    assert r1.n_reencoded == len(docs)
    assert r1.store.count() == len(docs)

    second = _FakeEmbedder()
    r2 = _build(tmp_path, docs, second)
    assert second.calls == [], f"二跑仍在编码：{second.calls}"
    assert r2.n_reencoded == 0

    # 三跑：多来一个新块，只能编码它自己
    third = _FakeEmbedder()
    _build(tmp_path, docs + [_doc("新来的", page=99)], third)
    assert third.total_encoded == 1, f"增量应只编码 1 条，实际 {third.total_encoded}"


# ---------------- ② search() 两条分支返回同型 ----------------
class _FakeReranker:
    """假精排：原序返回 (doc, 固定分)。真 reranker 在 CPU 上一条要 0.4-0.8s。"""

    def rerank(self, query, docs):
        return [(d, 1.0) for d in docs]


def test_search_returns_doc_score_pairs_in_both_modes(tmp_path):
    """开/关精排都必须返回 `[(Document, score)]`。

    这条直接钉住 `src/graph/nodes.py` 的依赖：`docs = [d for d, _ in hits]`。
    关精排时若返回裸 Document，整个问答链路会 ValueError —— 而且只在
    `RERANK_ENABLED=false` 时才炸，最容易被漏测的那一侧。
    """
    docs = [_doc(f"营业收入 第{i}页", page=i) for i in range(1, 9)]
    emb = _FakeEmbedder()

    from src.retrieval.hybrid import HybridRetriever

    for use_rerank in (False, True):
        kwargs = {"store": _store(tmp_path, f"contract{int(use_rerank)}")}
        if use_rerank:
            kwargs["reranker"] = _FakeReranker()
        r = HybridRetriever(docs, use_rerank=use_rerank, embedder=emb, **kwargs)
        hits = r.search("营业收入", top_k=3)

        assert len(hits) == 3, f"use_rerank={use_rerank} 返回条数不对：{len(hits)}"
        for item in hits:
            assert isinstance(item, tuple) and len(item) == 2, (
                f"use_rerank={use_rerank} 返回项不是 (doc, score)：{type(item)} {item!r}")
            doc, score = item
            assert isinstance(doc, Document) and isinstance(score, float)
