# Agentic RAG for Finance — 面向财报研报的智能问答系统

> 从零实现的金融文档 Agentic RAG：**Adaptive / Corrective / Self-RAG 三范式融合的 LangGraph 状态图**（手写 `add_node` / `add_conditional_edges`，非 prebuilt 封装）+ BM25/BGE-M3 混合检索 + CrossEncoder 精排 + 数值一致性幻觉拦截 + FastAPI SSE 流式。
>
> 全部指标由真实数据实测得出，见 [METRICS.md](METRICS.md)。

## 核心特性

| 模块 | 实现 | 实测结果 |
|---|---|---|
| 文档解析 | PyMuPDF `find_tables()` 表格结构化，页眉页脚清洗，bbox 去重 | 662 页真实年报 → 842 个表格块，数字零错位 |
| 分块 | 递归分块 + 表格整块保护（超长按行切分重复表头）+ 上下文前缀注入 | 10,014 分块 |
| 检索 | BM25(jieba+财报词典) + BGE-M3 dense → **加权 RRF** → BGE-Reranker-v2-m3 | Recall@5 **0.962** |
| Agent 图 | Adaptive（路由直答/单跳/多跳）+ Corrective（相关性过滤→改写→联网降级）+ Self-RAG（纯正则数值回溯校验→带失败提示重生成→兜底拒绝） | 陷阱拒绝率 0.75，幻觉自动重生成闭环 |
| 服务 | FastAPI + SSE（route/trace/status/token/citation/done 事件族）+ fakeredis 缓存 | 二次同问 26.9s → **0ms** |
| 前端 | 免构建玻璃拟态三栏：对话流（打字机）/ Agent 轨迹时间线 / 引用徽章 | 见 `web/index.html` |

**最有面试价值的发现**：等权 RRF 会被 BM25 长尾噪声稀释（dense 单路 0.923 → 等权融合 0.692），
加权 RRF（dense:bm25 = 0.7:0.3）恢复到 0.885，CrossEncoder 精排后 0.962。消融详见 METRICS.md。

## 快速开始

```bash
# 0) 环境：Python 3.12，GPU 可选（CPU 亦能跑，嵌入更慢）
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 1) 模型：BGE-M3 与 BGE-Reranker-v2-m3 放到 models/ 目录（huggingface.co/BAAI）
#    .env 里配置 LLM（DeepSeek 或任意 OpenAI 兼容网关）与 TAVILY_API_KEY，参考 .env.example

# 2) 放入年报 PDF 到 data/raw/（文件名约定：公司名_年份年年度报告.pdf）

# 3) 建索引（首次解析约 3-5 分钟，之后走语料缓存）
python scripts/ingest_real.py

# 4) 起服务（对话历史自动持久化：配了 MYSQL_URL 走 MySQL，否则落 data/db/chats.db）
python scripts/run_api.py          # http://127.0.0.1:8300

# 5) 浏览器打开 http://127.0.0.1:8300/web/（前端由服务直接托管），或直接调 API：
curl -N -X POST http://127.0.0.1:8300/api/chat -H "Content-Type: application/json" \
     -d '{"question":"贵州茅台2024年营业收入是多少","session_id":"demo"}'
```

前端除对话/轨迹/引用外还接了：**「＋ 上传PDF」入库**（`/api/ingest/upload`，实时显示分块进度）、**刷新后历史对话自动恢复**（localStorage 会话 ID + `/api/history`）、**存储层运维条**（`/api/store/stats`：MySQL/SQLite、轮次、会话数、最近写入）。

### Docker 一键部署（api + mysql + redis）

```bash
export AGNES_API_KEY=...            # 或 DEEPSEEK_API_KEY / TAVILY_API_KEY
docker compose up -d                # api:8300 + mysql:8.4 + redis:7
# 对话落 MySQL(rag 库)、高频问答缓存走 Redis；models/ 与 data/ 卷挂载不进镜像
```

存储层说明：会话历史经 SQLAlchemy URL 抽象双引擎——`MYSQL_URL` 配了连 MySQL（生产形态，
`chat_turns` 表），缺省落 SQLite（`data/db/chats.db`，开发零配置），`/api/history` 语义不变；
高频问答缓存 `REDIS_URL` 配了走真实 Redis，缺省降级 fakeredis（进程内，接口一致）。
运维视图：`GET /api/store/stats` 返回后端类型 / 总轮次 / 会话数。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | SSE 流式问答（事件：route / trace / status / token / citation / done / error） |
| POST | `/api/ingest` | 指定 PDF 路径入库并热重建索引 |
| POST | `/api/ingest/upload` | multipart 上传 PDF |
| GET | `/api/documents` | 已入库文档统计 |
| GET | `/api/history/{session_id}` | 会话历史（多轮追问靠 thread_id 串起） |
| GET | `/health` | 健康检查 |

## 评测

```bash
python evals/build_goldset.py        # 构建金标（答案子串必须在指定页原文出现，自检通过才产出）
python evals/run_retrieval_eval.py   # 四通路 Recall@5 消融 → evals/retrieval_result.json
python evals/run_trap_eval.py        # 诱导幻觉陷阱拒绝率 → evals/trap_result.json
```

评测集：26 条检索项（茅台/宁德/比亚迪主要会计数据 + 定性落页）+ 4 条诱导幻觉陷阱。

## 项目结构

```
src/
├── core/        config（pydantic-settings，路径全部锚定项目内）/ llm 多提供方工厂
├── ingestion/   pdf_parser（PyMuPDF 表格结构化）/ chunker（表格保护 + 上下文前缀）
├── retrieval/   bm25 / embeddings(BGE-M3 dense+sparse) / vector_store(Chroma) / reranker / hybrid(加权 RRF)
├── graph/       state(TypedDict+reducer) / nodes(路由·过滤·改写·联网·生成·数值校验) / rag_graph(手写状态图)
└── api/         engine(单例+语料缓存) / app(FastAPI+SSE) / schemas
evals/           金标构建 + 检索消融 + 陷阱评测
web/             免构建玻璃拟态前端
scripts/         建索引 / 起服务 / 冒烟测试
```

## 设计取舍（为什么不用现成封装）

- **手写 LangGraph 状态图**而非 `create_react_agent`：RAG 流水线是工作流不是自由 Agent，
  显式状态字段（rewrite_count / retrieve_rounds / regen_count）是防死循环的关键，
  条件边映射让每一条自纠正路径都可解释、可画图。
- **数值校验用纯正则回溯**而非 LLM 自评：金融场景数值错报是事故级风险，规则法零成本、零漏判、可单测。
- **加权 RRF** 而非等权：见消融数据。权重可配置（`rrf_dense_weight` / `rrf_bm25_weight`）。

## License

MIT
