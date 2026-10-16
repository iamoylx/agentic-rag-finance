# Agentic RAG for Finance — 面向财报研报的智能问答系统

> 从零实现的金融文档 Agentic RAG：**Adaptive / Corrective / Self-RAG 三范式融合的 LangGraph 状态图**（手写 `add_node` / `add_conditional_edges`，非 prebuilt 封装）+ BM25/BGE-M3 混合检索 + CrossEncoder 精排 + 数值一致性幻觉拦截 + FastAPI SSE 流式。
>
> 全部指标由真实数据实测得出，见 [METRICS.md](METRICS.md)。

## 核心特性

| 模块 | 实现 | 实测结果 |
|---|---|---|
| 文档解析 | PyMuPDF `find_tables()` 表格结构化，页眉页脚清洗，bbox 去重 | 662 页真实年报 → 842 个表格块，数字零错位 |
| 分块 | 递归分块 + 表格整块保护（超长按行切分重复表头）+ 上下文前缀注入 | 10,014 分块 |
| 检索 | BM25(jieba+财报词典) + BGE-M3 dense → **加权 RRF** → BGE-Reranker-v2-m3 | Recall@5 **0.962** / MRR@5 0.686 / nDCG@5 0.814（四通路消融见 METRICS） |
| Agent 图 | Adaptive（路由直答/单跳/多跳）+ Corrective（相关性过滤→改写→联网降级）+ Self-RAG（纯正则数值回溯校验→带失败提示重生成→兜底拒绝） | 陷阱拒绝率 0.75，幻觉自动重生成闭环 |
| **流式数值闸门** | token 级扣留「正在拼装中的数字」，凑齐即校验、通过才放行；**未校验数字永不上屏**（`src/safety/number_gate.py`） | 单帧开销 **1.0 μs**、首字延迟**零变化**；编造数字零字符上屏（真浏览器实测，见图） |
| 服务 | FastAPI + SSE（route/trace/status/token/citation/**reset**/done 事件族）+ fakeredis 缓存 | 二次同问 26.9s → **0ms** |
| 前端 | 免构建玻璃拟态三栏：对话流（打字机）/ Agent 轨迹时间线 / 引用徽章 | 见 `web/index.html` |
| 溯源 | 答案带页码引用 + 引用有效性扫描（语料构建「有效页集合」） | 引用有效率 **0.885**（防幻觉引用）/ 严格命中金标页 0.500（页码多义性分析见 METRICS） |

**最有面试价值的发现**：等权 RRF 会被 BM25 长尾噪声稀释（dense 单路 0.923 → 等权融合 0.692），
加权 RRF（dense:bm25 = 0.7:0.3）恢复到 0.885，CrossEncoder 精排后 0.962；但 MRR@5/nDCG@5
揭示精排优化的是 top-5 覆盖而非 top-1 精度（dense_only MRR 0.735 > 精排 0.686）——
指标越细，优化方向越明确。消融与引用双口径详见 METRICS.md。

## 流式数值显示闸门：把「校验」从生成之后提前到生成之中

项目的承诺是「答案里的数字都能在检索到的原文里找到」。这个承诺在**流式输出**下有真实矛盾：
校验必须等答案生成完，而答案是一边生成一边往屏幕上推的。常见做法是二选一——
要么等校验完再出字（牺牲流式），要么先出字、错了再撤回（错误内容**已经到过用户眼前**）。

这里用的是第三条路：**只扣住「正在拼装中的数字」**。模型的 token 流每来一段，
就把尾巴上「可能还没拼完的数字」留在缓冲里，一旦这个数字后面跟了非数字字符（= 它定型了），
**立刻**判定它是否能在语料里找到，通过才放行。正文（无数字的部分）照常逐字流式。

它成立靠一条不变式：

> **放行边界永远不许落在任何一个数字 token 的内部。**

| 为什么不用另外两个方案 | |
|---|---|
| 先流后撤 | 达成度 **0**——错误数字确实上过屏（截图/录屏/人眼已捕捉）。金融场景里「看到过」本身就是风险 |
| 整段扣留到校验完 | 达成度 1，但让 **100% 的请求**都付「首字延迟 = 整段生成」；而校验失败的异常路径实测只占 **4/26 ≈ 15%**——为少数场景让所有人买单 |

实测代价（`evals/bench_gate_overhead.py`，20 条真实答案、84 个数字 token）：

| 项 | 实测 |
|---|---|
| 单帧处理开销 | 中位 **1.00 μs** / 最大 1.26 μs（相对端到端 28.4s，占比 < 0.0001%） |
| 首字延迟 | **完全一致**（首块即放行，与无闸门基线逐块相同） |
| 额外扣留时长 | 只多扣「1 个数字」拼完的长度：中位 3 / p90 11 / 最大 22 字符 |

真实浏览器验证（编造数字被扣 → 两次撤回自纠 → 第三次校验通过）：

| 第一次撤回 | 第二次撤回 | 校验通过 |
|---|---|---|
| ![reset1](assets/gate-reset-1.png) | ![reset2](assets/gate-reset-2.png) | ![passed](assets/gate-passed.png) |

完整前端（Agent 轨迹时间线 + 页码引用徽章）：`assets/frontend-trace.png`。
设计权衡与方案支配关系分析见 **[DESIGN-streaming-gate.md](DESIGN-streaming-gate.md)**。

> 边界（不吹）：它只保护**能被规则判定、且比对语料提前可得**的字段（本项目是数值）。
> 准确表述是「杜绝数值**编造**」，不是「保证答案正确」——它拦不住「选错但真实存在」的数字。

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
| POST | `/api/chat` | SSE 流式问答（事件：route / trace / status / token / citation / **reset** / done / error） |
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

## 测试

```bash
python -m pytest -q            # 59 passed（testpaths=tests，约 31s；一次性的模型导入占 ~19s）
python evals/bench_gate_overhead.py   # 闸门性能实测（零 LLM 消耗）
```

| 文件 | 用例 | 验的是什么 |
|---|---|---|
| `tests/test_number_gate.py` | 14 | 闸门与数字规则的单元行为（token 边界、全角逗号、子串语义护栏） |
| `tests/test_gate_property.py` | 39 | **性质测试**：不变式 I1–I4 + 生成的各种切法（穷举短答案 / 采样长答案） |
| `tests/test_sse_gate.py` | 5 | SSE 协议层：假 stream 喂进 `_chat_gen`，验翻译逻辑 |
| `tests/test_store.py` | 1 | 存储层 SQLite 回退路径 |

**断言不是写完就算数**——每条都做了**变异验证**（把被测代码按「只改一件事」的方式破坏，
看哪条断言会响），实测矩阵见 `tests/test_gate_property.py` 模块文档。要点：

| 变异（只改一件事） | I1 前缀 | I2 边界 | I3 屏幕数字 | I4a 判定一致 |
|---|---|---|---|---|
| M1 旧正则 `\.\d+`（真实发生过的 bug） | 0 | **7** | 0 | 0 |
| M2 判定脱钩（放行序列逐字节不变，只改结论） | 0 | 0 | 0 | **5** |
| M3 闸门旁路（所有内容原样上屏） | 0 | 10 | **5** | 5 |
| M4 放行路径清洗 `safe.replace(",", "")` | **5** | 5 | 0 | 0 |

每条断言都至少被一种变异触发 → 没有恒真的空转断言。两个反直觉发现值得记：
① 旧正则那个 bug **只有 I2 能抓**——它是**显示时序**缺陷（最终文本其实拼完整了），
不是结论缺陷，所以「数字在不在语料里」(I3) 和「数字有没有被打断」(I2) 是正交的；
② 同一测试内部有多层 assert 时，失败信息只能证明「第一个 assert」失败 —— 差分测试因此
拆出不断言的 `_drive_raw`，否则会「以差分失败的名义」报红而实际报警的是边界断言。

## 项目结构

> 整体 pipeline 图、分层调用关系、逐文件阅读指南与术语表见 **[ARCHITECTURE.md](ARCHITECTURE.md)**。

```
src/
├── core/        config（pydantic-settings，路径全部锚定项目内）/ llm 多提供方工厂
├── ingestion/   pdf_parser（PyMuPDF 表格结构化）/ chunker（表格保护 + 上下文前缀）
├── retrieval/   bm25 / embeddings(BGE-M3 dense+sparse) / vector_store(Chroma) / reranker / hybrid(加权 RRF)
├── graph/       state(TypedDict+reducer) / nodes(路由·过滤·改写·联网·生成·数值校验) / rag_graph(手写状态图)
├── safety/      number_rules（零依赖叶子模块：数字提取/归一化/子串比对，图节点与闸门共用）
│                number_gate（流式数值显示闸门：token 级扣留 + 凑齐即校验）
└── api/         engine(单例+语料缓存) / app(FastAPI+SSE) / schemas
evals/           金标构建 + 检索消融 + 陷阱评测 + 权重扫描 + 闸门性能实测
web/             免构建玻璃拟态前端
assets/          README 用验证截图（真浏览器实测）
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
