# agentic-rag-finance 架构梳理与代码阅读指南

> 本文是 P1 的「地图」：先看流程图建立整体心智模型，再按第 6 节顺序逐文件精读。
> 文档、代码注释、README 三者口径一致——同一概念在任何一处出现，含义不变（术语表见第 7 节）。
> （姊妹项目 text2sql-agent 有同款 ARCHITECTURE.md，两项目共用「SQLAlchemy URL 抽象」等工程模式。）

---

## 1. 一图总览：数据入库与一个请求的一生

```
【离线/启动期】语料构建（engine.build_corpus，带指纹缓存）

 data/raw/*.pdf（3 家公司真实年报 662 页）
        │
        ▼
 AnnualReportParser（PyMuPDF）
   表格：find_tables() 检测+提取同引擎 → Markdown（数字零错位）
   文本：按块提取 → 剔除表格区域（bbox 重叠比）→ 清洗页眉页脚
   ★页码贯穿全程：引用溯源与数值校验的命脉
        │
        ▼
 FinancialChunker
   文本：递归分块（512 字 / 64 重叠，中文分隔符优先级）
   表格：整表单块；超长按行切 + 每片重复表头
   每块注入上下文前缀【公司 | 年份 | 第N页】
        │
        ▼  语料缓存 data/parsed/corpus.jsonl（首行 fingerprint=文件名+mtime）
        │
 HybridRetriever._build
   路1 BM25 内存索引（jieba + 财报自定义词典）
   路2 BGE-M3 dense → Chroma 持久化（md5 幂等 upsert）
   精排器 BGE-Reranker-v2-m3（单例常驻）

【在线期】一个提问的一生

 用户提问 ──► POST /api/chat (SSE)
        │
        ▼
 ┌─ 缓存快路：Redis/fakeredis 命中（会话首轮才查）→ 逐帧补发答案
 │
 └─ 慢路：graph.astream(stream_mode=["updates","messages"]) 双流
        │
        ▼
 route_query（Adaptive RAG：direct / single / multi + 指代消解 standalone）
   ├─ direct ──► direct_answer ──────────────────────────► END
   └─ single/multi ──► retrieve（双路 Top-20 → 加权RRF → 精排 Top-5）
                          │
                          ▼
                    grade_documents（Corrective RAG：LLM 逐块判相关性）
        ┌────────────┼──────────────────┐
        │有相关文档    │无相关且可改写      │无相关且改写用尽
        ▼            ▼                  ▼
     generate ◄── rewrite_query ──► retrieve（改写回环）
        │            （≤2 次）   web_search（Tavily 联网降级）──► generate
        ▼
 check_hallucination（Self-RAG：纯规则数值回溯，答案数字必须能在原文找到）
        │
   ┌────┴─────────┐
   │通过           │未通过（≤2 次重生成，带失败数字提示）
   ▼              ▼
 finalize ──► END          fallback_answer（降级收尾）──► END

 生成侧护栏：答非所问禁令 / 禁止单位换算 / 页码引用 [引用: 第X页] / 空响应重试
        │
        ▼
 双流翻译成 SSE：route / status / trace / token(打字机) / citation / done / error
        │
        ▼
 web/index.html 三栏前端 ＋ store.py 落库 chat_turns（MySQL/SQLite）
```

## 2. 分层调用关系与数据流向

```
调用方向（上层依赖下层，下层不知道上层存在）：

  web/index.html ── HTTP/SSE ──► api/app.py ──► graph/rag_graph.py（LangGraph）
                                    │                    │
                     api/engine.py（引擎单例）        graph/nodes.py（9 节点）
                       │        │                        │
        api/store.py   │   api/schemas.py            retrieval/hybrid.py
     (chat_turns 持久化)│                          （唯一检索入口）
                       │                                │
                       ▼                 ┌──────────────┼───────────────┐
              ingestion/*（解析+分块）      ▼              ▼               ▼
                       │             bm25.py    vector_store.py   reranker.py
                       ▼                 │           ▲                │
              core/config.py ◄── 全部 ──┘    embeddings.py（BGE-M3 单例）◄┘
                                        (jieba 词典)      (dense/sparse)

数据流三句话：
- 入库向：PDF → Block(带页码) → Document(带前缀与 metadata) → BM25 索引 + Chroma 向量。
- 查询向：question → route/standalone → retriever.search() → documents → generate →
  answer+citations → SSE。
- 校验向：answer 中的数字 ↔ documents 原文数字（归一化后子串匹配）→ pass/regen/fallback。
```

## 3. 三范式融合（核心设计，也是面试叙事主线）

| 范式 | 回答的问题 | 落点节点 | 防失控 |
|---|---|---|---|
| **Adaptive RAG** | 这个问题需要检索吗？ | route_query（direct/single/multi 三路） | 解析失败保守回落 single（宁可多检索） |
| **Corrective RAG** | 检索结果质量够吗？ | grade_documents → rewrite_query（≤2）→ web_search 兜底 | rewrite_count / retrieve_rounds / web_used |
| **Self-RAG** | 生成的答案可信吗？ | check_hallucination（**纯规则**数值回溯，零 LLM 成本）→ 失败带提示重生成（≤2）→ fallback 拦截 | regen_count |

三者在**同一张手写 StateGraph**（`add_node` / `add_conditional_edges`，禁用 create_react_agent）里协作——这是简历「三范式融合」的准确含义：不是三个系统，是一张图上的三组节点与条件边。

## 4. 检索流水线（P1 最重的工程细节）

```
query ──┬─► BM25（jieba 精确模式 + 财报词典强制成词）───── Top-20 ─┐
        │                                                        ├─► 加权 RRF
        └─► BGE-M3 dense（[CLS] 归一化 1024 维，Chroma cosine）── Top-20 ─┘
                                weights = bm25 0.3 / dense 0.7, k=60
        ▼
 融合 Top-20 ──► BGE-Reranker-v2-m3（CrossEncoder 逐对交互）──► Top-5 进 prompt
```

- **为什么加权**：等权 RRF 会被 BM25 长尾噪声稀释（Recall@5 0.923→0.692，消融实测），
  dense:bm25 = 0.7:0.3 恢复到 0.885，精排后 0.962。
- **诚实 trade-off（S8 复盘补测）**：精排满血 Recall@5 但 MRR@5 低于纯 dense——
  reranker 优化「top-5 覆盖」而非「top-1 精度」，消融见 evals/retrieval_result.json。
- **显存预算**：embedder + reranker 各 ~2GB 常驻（单例），4070 8GB 够用；
  sparse 投影中间张量巨大，encode_sparse 强制 batch=1。

## 5. 生成与校验的护栏细节（面试常追问）

| 护栏 | 实现 | 为什么 |
|---|---|---|
| 答非所问禁令 | GEN_PROMPT 硬规则（问 A 只有 B → 必须回「未找到」） | 同页近似指标会诱导模型输出 B，评测陷阱题的教训 |
| 禁止单位换算 | 逐字引用原文数字（170,899,152,276.34 元 原样） | 换算引入精度误差，且破坏数值回溯匹配 |
| 页码引用 | 片段头带页码，答案尾 [引用: 第X页]；网络块标 [来源: 网络] | 金融场景可溯源是底线；page 字段从解析层一路带过来 |
| 数值回溯校验 | 纯正则：答案数字（去千分位）必须能在引用原文找到 | 零 LLM 成本、可单测、可解释；幻觉拦截 4/4 的来源 |
| 空响应兜底 | 网关偶发空 content → 重试一次 → 仍空按「未找到」收尾 | 空串会绕过数值校验被误判通过 |
| 多轮指代消解 | route 节点把「那2024年呢」改写成 standalone | checkpointer thread_id 串会话，history 进 prompt |

## 6. 代码阅读顺序指南

> 原则：**先配置后模型、先离线入库后在线检索、先节点后组图、先引擎后接口**。
> 每个文件给出：有什么内容 / 起什么作用 / 与其他模块的关系 / 读时注意的坑。

### 第 1 层 · 配置与模型

**① `src/core/config.py`（113 行）**
- 内容：Settings——LLM 双提供方、Tavily、BGE-M3/Reranker 本地模型路径与显存参数、检索超参（chunk/RRF 权重/各循环上限）、Redis/MySQL URL、项目内路径推导。
- 作用：全部可变项唯一入口；`rrf_dense_weight=0.7` 这类消融结论直接沉淀在注释里。
- 关系：所有模块 import；`ensure_dirs()` 启动时建目录。
- 注意：路径全部锚定 PROJECT_ROOT（不写 C 盘用户目录）；`max_regen/max_rewrite/max_retrieve_rounds` 三个上限是图条件边的判停依据。

**② `src/core/llm.py`（68 行）**
- 内容：`get_llm()`（lru_cache 单例）+ `get_cheap_llm()`（短输出分类任务用）+ `_resolve_provider()`。
- 作用：全项目唯一 LLM 出口；`streaming=True` 是 SSE 打字机的前提。
- 关系：graph 节点按任务选贵/便宜两档（路由/打分/改写用 cheap，生成用 full）。
- 注意：Agnes 对 max_tokens 敏感——过小（如 16）返回**空 content 而不报错**，cheap 档最低给 1024。

### 第 2 层 · 入库层（离线，决定检索上限）

**③ `src/ingestion/pdf_parser.py`（207 行）**
- 内容：`AnnualReportParser.parse()`——逐页表格提取（PyMuPDF find_tables→Markdown）+ 文本块提取（bbox 剔除表格区域）+ 页眉页脚清洗 + `Block` 中间表示。
- 作用：数据源的唯一入口；「页码命脉」从这里开始。
- 关系：输出 Block 给 chunker；`source` 文件名进 metadata 供后续引用与去重。
- 注意：表格提取选 PyMuPDF 而非 pdfplumber 是踩坑决策（pdfplumber 重建单元格会把长数字切碎）；`_in_any_zone` 用「交集面积/文本块面积 ≥0.3」防表格内容双份入库。

**④ `src/ingestion/chunker.py`（157 行）**
- 内容：`FinancialChunker`——文本递归分块 + 表格整块保护（超长按行切、每片带表头）+ 上下文前缀注入 + `summarize_chunks` 自检。
-作用：分块质量决定 Recall@5 上限；前缀【公司|年份|页码】解决「三家公司的营业收入块向量几乎相同」。
- 关系：吃 Block 产 Document；`metadata.page/source/block_type` 一路带到引用与校验层。
- 注意：`table_partial: True` 标记超长表格的切片；中文按「字」计数（length_function=len）无需 tokenizer。

### 第 3 层 · 检索层（四件套 + 一个融合器）

**⑤ `src/retrieval/bm25.py`（114 行）**
- 内容：jieba 分词（财报词典 `load_finance_dict` 强制成词）+ BM25Okapi 索引与查询。
- 作用：稀疏通路——证券代码、精确时间短语、科目名的字面匹配。
- 关系：hybrid 双路之一；`tokenize()` 必须索引/查询两侧同源。
- 注意：分词质量是 BM25 生死线——「归属于母公司所有者的净利润」这类长术语不进词典就会被切碎。

**⑥ `src/retrieval/embeddings.py`（132 行）**
- 内容：`BGEM3Embedder` 单例——dense（[CLS] 归一化）与 sparse（sparse_linear.pt 投影，log(1+relu) 按 token 取 max）双编码。
- 作用：稠密通路向量来源；用 transformers 手写避免 FlagEmbedding 重依赖。
- 关系：hybrid 与 vector_store 之间唯一算向量的地方（存/算解耦，换模型只重建索引）。
- 注意：sparse 中间张量 (batch, seq, 250002) 约 6GB → 强制 batch=1；显存不足自动落 CPU（fp16 仅 CUDA）。

**⑦ `src/retrieval/vector_store.py`（118 行）**
- 内容：`ChromaVectorStore`——md5(source|page|content) 幂等 upsert（分批 4096）+ cosine 检索 + metadata 标量化。
- 作用：dense 向量持久化；id 稳定 = 重复入库不翻倍。
- 关系：embedder 算好向量传入；hybrid 持有它做 search_dense。
- 注意：Chroma distance ∈[0,2]，返回前换算 similarity=1-distance；真实年报存在同页同内容重复块，upsert 前按 id 合并。

**⑧ `src/retrieval/reranker.py`（81 行）**
- 内容：`BGEReranker` 单例——CrossEncoder 对 (query, doc) 对逐对打分排序。
- 作用：粗排（排名融合）→ 精排（充分交互）的最后一步，把顺序纠正成真正相关性序。
- 关系：hybrid 的可选末级（use_rerank=False 可只测召回）。
- 注意：必须用 AutoModelForSequenceClassification 加载才有 logits；分数是无界 logit 只比相对大小。

**⑨ `src/retrieval/hybrid.py`（126 行）**
- 内容：`rrf_fuse`（支持加权的 RRF）+ `HybridRetriever`（建双索引 + `search()` 完整流水线 + 三个消融钩子 `search_dense_only` / `search_bm25_only` / `search_hybrid_no_rerank`）。
- 作用：graph 层唯一检索入口——上层完全不感知 BM25/Chroma/Reranker 细节。
- 关系：评测脚本 run_retrieval_eval.py 的四通路对比就是靠三个钩子实现的。
- 注意：RRF 只看排名不看分数（免归一化）；加权动机与数据见 METRICS——这是消融驱动调参的标准案例。

### 第 4 层 · 图层（三范式融合的手写状态图）

**⑩ `src/graph/state.py`（48 行）**
- 内容：`RAGState` TypedDict——输入/路由/检索中间态/生成校验/收尾五组字段 + `trace`（Annotated operator.add 追加合并）。
- 作用：节点间的数据契约；刻意**不用 add_messages**——这是工作流不是对话流，显式字段更清晰。
- 关系：rag_graph 以它为 schema；三个计数器字段是条件边判停依据。
- 注意：trace 的写入方式统一为「节点在增量 dict 里带 trace: [msg]」，不要原地 mutate（channel 合并以返回值为准）。

**⑪ `src/graph/nodes.py`（322 行）**
- 内容：9 个节点函数——route_query（路由+指代消解）/ direct_answer / retrieve（闭包工厂注入 retriever）/ grade_documents / rewrite_query / web_search / generate（护栏最密集处）/ check_hallucination（纯规则）/ fallback_answer。
- 作用：三范式的全部业务逻辑；`_parse_json` 宽容解析是「LLM 输出一律 JSON」约定的兜底。
- 关系：被 rag_graph 注册；retriever 由 `make_retrieve_node` 闭包注入（节点保持纯函数）。
- 注意：generate 的三条铁律（答非所问禁令/禁止单位换算/页码引用）+ 失败数字回喂重生成 + 空响应兜底，全是评测踩坑换来的；grade 解析失败视为相关（保守方向：宁可不删）。

**⑫ `src/graph/rag_graph.py`（119 行）**
- 内容：`build_rag_graph()`——9 节点连线 + 4 个条件边（路由分流/检索后分流/校验后分流）+ `answer()` 入口。
- 作用：把三范式融进一张图；对照图读懂每个条件边的四个去向。
- 关系：engine._build 调它产图；checkpointer 传入以支持多轮。
- 注意：`after_grade` 的优先级顺序 = 改写 → 联网 → 降级（降级是最后手段）；finalize 节点只做透传保持图逻辑清晰。

### 第 5 层 · 服务层（单例、缓存、持久化、SSE）

**⑬ `src/api/engine.py`（154 行）**
- 内容：`Engine` 单例——startup 冷建库 / build_corpus（**指纹缓存**：文件名+mtime 变了才重解析）/ ingest 增量重建（锁挡并发）/ Redis-fakeredis 高频缓存。
- 作用：连接入库层与图层的枢纽；模型加载 5-6s 绝不能按请求加载。
- 关系：app.py 的 lifespan 调 startup；ingest 后 `_build` 整体重建（retrieve 节点闭包持有旧 retriever，必须换新图）。
- 注意：缓存只收「有真实依据的成功回答」（有引用且非缓存命中）；cache_key 是去空白小写后的 md5。

**⑭ `src/api/store.py`（133 行）**
- 内容：`chat_turns` 持久化——MySQL（MYSQL_URL）/SQLite 双引擎、ping 探活、append_turn / load_history / stats。
- 作用：简历「MySQL 存对话」落点；替代进程内 dict，跨重启可用。
- 关系：与 P2 的 connect.py 同一「SQLAlchemy URL 抽象」工程模式，两项目互相印证。
- 注意：DDL 双方言（AUTO_INCREMENT vs AUTOINCREMENT）；created_at 统一转 epoch 秒屏蔽两库类型差异。

**⑮ `src/api/schemas.py`（40 行）**
- 内容：ChatRequest / IngestRequest / Citation / HealthResponse。
- 作用：HTTP 契约集中一处，路由函数只编排不校验。
- 关系：session_id 即 checkpointer 的 thread_id（多轮追问的钥匙）。

**⑯ `src/api/app.py`（247 行）**
- 内容：lifespan 冷启动 + `/api/chat` SSE（缓存快路 / 双流慢路）+ ingest（含 multipart 上传）/ documents / history / store/stats / health 路由。
- 作用：图的 HTTP 外壳——把 updates+messages 双内部流翻译成 7 类 SSE 事件。
- 关系：_ANSWER_NODES 白名单过滤 token 流（路由/打分节点的 LLM 输出不得混进答案打字机）；ingest 走 engine 并触发重建。
- 注意：`stream_mode=["updates","messages"]` 双流是「中间过程可视化」的技术底座；sent_tokens==0 时兜底 get_state 补发（个别网关不支持流式回调）；**前端 SSE 帧必须按 /\r?\n\r?\n/ 切**（sse-starlette 发 CRLF——本仓库踩过的最隐蔽 bug）。

### 第 6 层 · 前端与外围

**⑰ `web/index.html`（350 行）**
- 内容：免构建三栏玻璃拟态前端——对话流（打字机+轨迹 chips）/ 引用徽章 / 存储运维条 / PDF 上传 / 会话恢复。
- 作用：演示与验收入口；7 类 SSE 事件的最终消费者。
- 注意：localStorage 存 session_id 实现刷新恢复；前端改动必须真浏览器截图验收（curl 正确 ≠ DOM 正确，本项目与 P2 各踩过一次）。

**⑱ `scripts/`（6 个）**
- `run_api.py` 启动 8300；`ingest_real.py` 真实年报入库；`verify_models.py` 模型自检；`mini_rag.py` / `smoke_test.py` / `test_hybrid.py` / `test_graph.py` 是分阶段冒烟脚本（读代码时可当最小可运行示例用）。

**⑲ `evals/`**
- `build_goldset.py`（26 条金标构建+自检）/ `run_retrieval_eval.py`（四通路 R@5/MRR@5/nDCG@5）/ `run_e2e_eval.py`（26 条端到端）/ `run_trap_eval.py`（陷阱拒绝率）。**读评测脚本是最好的「指标口径」教材**；结果 json 与 METRICS.md 一一对应。

**⑳ `tests/test_store.py`**
- 存储层单测（SQLite 回退路径）；跑全量：`python -m pytest -q`。

## 7. 术语表（与代码注释同口径）

| 术语 | 定义 | 落点 |
|---|---|---|
| Adaptive RAG | 按问题类型路由：direct 直答 / single 单跳 / multi 多跳 | nodes.route_query |
| Corrective RAG | 检索质量评估 → 改写回环 → 联网降级的三级自纠 | grade/rewrite/web_search |
| Self-RAG | 生成后数值回溯校验（纯规则），失败带提示重生成 | check_hallucination |
| 加权 RRF | score=Σ w·1/(k+rank)，dense:bm25=0.7:0.3（消融驱动） | hybrid.rrf_fuse |
| 答非所问禁令 | 问 A 片段只有 B 时必须回「未找到」，禁止列举 B | nodes.GEN_PROMPT 规则 3 |
| 数值回溯校验 | 答案数字（去千分位）必须能在引用原文找到 | nodes.check_hallucination |
| 页码命脉 | page 字段从解析层贯穿到引用/校验/前端跳转 | pdf_parser.Block.page |
| 指纹缓存 | 语料 jsonl 首行存 md5(文件名+mtime)，命中跳过数分钟解析 | engine.build_corpus |
| 双流翻译 | astream(updates+messages) → route/trace/status + token 打字机 | app._chat_gen |
| 引用双口径（评测） | 严格命中金标页 0.500 / 引用页确含该数值 0.885 | evals/run_citation_eval.py |
