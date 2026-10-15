"""
FastAPI 服务层（P1-S6）：SSE 流式问答 + 入库 + 会话历史 + 健康检查。

语法：FastAPI lifespan 管理引擎生命周期；sse-starlette 的 EventSourceResponse
     把 async generator 的产出逐条推给浏览器（text/event-stream）。
原理：
  - 流式打字机的关键在 graph.astream(stream_mode=["updates","messages"])：
    「updates」给出每个节点返回的状态增量（路由决策 / 检索轮次 / 校验结果），
    「messages」给出 LLM 的 token 增量（靠 on_llm_new_token 回调），
    两条流合并翻译成 SSE 事件：route / status / trace / token / citation / done / error。
  - 把中间过程可视化是演示杀手锏——面试官能看到 Agent 在思考、检索、自我纠正。
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from src.api import store
from src.api.engine import engine
from src.api.schemas import ChatRequest, HealthResponse, IngestRequest
from src.safety.number_gate import NumericHoldBackGate
from src.safety.number_rules import build_corpus_norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from src.core.config import settings

# 会产生「最终答案」token 的节点；路由/打分/改写节点的 LLM 输出不应混进答案流
_ANSWER_NODES = {"generate", "direct_answer", "fallback_answer"}

# 需要过「数值显示闸门」的节点——只此一个，有明确依据：
#   check_hallucination 只挂在 generate 之后（direct 分支直接到 END，不校验），
#   所以只有 generate 的输出承诺了「数字都经过校验」。给 direct_answer 也加闸门
#   会「过度拦截」：它本就没有语料可比对，任何数字都会被判编造，
#   屏幕上会莫名其妙少内容，而权威校验其实从未对它提出过这个要求。
_GATED_NODES = {"generate"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时冷加载模型与索引（约 6-10 秒），关闭时无需清理。"""
    if not engine.ready:
        await asyncio.to_thread(engine.startup)   # to_thread：别阻塞事件循环
    # 存储层探活：MYSQL_URL 配了走 MySQL，否则落 SQLite（结果写日志）
    info = await asyncio.to_thread(store.ping)
    print(f"[store] chat history backend = {info['backend']}")
    yield


app = FastAPI(title="Agentic RAG for Finance", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 开发期放开；上生产改白名单
    allow_methods=["*"],
    allow_headers=["*"],
)
# 前端挂载：/web 直接由本服务托管（免构建静态页），浏览器访问 http://127.0.0.1:8300/web/
app.mount("/web", StaticFiles(directory=PROJECT_ROOT / "web", html=True), name="web")


def _sse(event: str, data: dict) -> dict:
    """统一 SSE 帧格式；ensure_ascii=False 保证中文原样传输。"""
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


def _citations(final_state: dict) -> list[dict]:
    """从最终 state 的 documents 提取去重引用；年报块带页码，网络来源块标记 page=-1。"""
    seen, out = set(), []
    for d in final_state.get("documents", []):
        is_web = d.metadata.get("block_type") == "web"
        key = ("web", -1) if is_web else (d.metadata.get("source", ""), d.metadata.get("page", -1))
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": "web" if is_web else key[0], "page": key[1],
                    "snippet": d.page_content[:80]})
    return out


async def _chat_gen(req: ChatRequest):
    """SSE 主生成器：缓存命中走快路，否则跑图并把两条内部流翻译成 SSE。"""
    t0 = time.time()
    thread_id = req.session_id

    # ---------- 快路：高频问题缓存（会话首轮才查，避免多轮追问误命中）----------
    history = await asyncio.to_thread(store.load_history, thread_id)
    fresh_session = not history
    cached = engine.cache_get(req.question) if fresh_session else None
    if cached:
        yield _sse("route", {"route": "cache", "query": req.question})
        for piece in cached["citations"]:
            yield _sse("citation", piece)
        for chunk in _split_tokens(cached["answer"]):
            yield _sse("token", {"t": chunk})
        yield _sse("done", {"answer": cached["answer"], "cached": True,
                            "elapsed_ms": int((time.time() - t0) * 1000)})
        await asyncio.to_thread(store.append_turn,
                                thread_id, req.question, cached["answer"], 0, True)
        return

    # ---------- 慢路：完整跑图 ----------
    init_state = {"question": req.question, "query_used": req.question,
                  "history": history,
                  "retrieve_rounds": 0, "rewrite_count": 0, "regen_count": 0,
                  "web_used": False, "trace": []}
    config = {"configurable": {"thread_id": thread_id}}
    sent_tokens, answer_text, final_state = 0, "", {}
    # 数值显示闸门的两个状态：比对语料 + 闸门实例本身。
    # corpus_norm 只能等检索结果到达才拿得到（它来自 state["documents"]），
    # 所以闸门必须**延迟创建**——这就是「要控制这个 stream 才能做 hold-back」的具体含义。
    corpus_norm = ""
    gate: NumericHoldBackGate | None = None

    try:
        stream = engine.graph.astream(init_state, config=config,
                                      stream_mode=["updates", "messages"])
        async for mode, payload in stream:
            if mode == "updates":
                # payload 形如 {节点名: 该节点返回的状态增量}
                for node, update in payload.items():
                    for line in update.get("trace", []):
                        yield _sse("trace", {"node": node, "msg": line})
                    # retrieve / grade_documents / web_search 都可能在 generate 之前设 documents。
                    # 一旦语料（或它带来的检索结果）变了，旧闸门的比对母本即作废 → 重建。
                    # 此刻还在出 token 之前，所以置空不会丢已扣留的内容。
                    if "documents" in update:
                        corpus_norm = build_corpus_norm(update["documents"])
                        gate = None
                    if node == "route_query" and "route" in update:
                        yield _sse("route", {"route": update["route"],
                                             "query": update.get("query_used", req.question)})
                    elif node == "retrieve":
                        yield _sse("status", {"node": "retrieve",
                                              "round": update.get("retrieve_rounds"),
                                              "hits": len(update.get("documents", []))})
                    elif node == "grade_documents":
                        yield _sse("status", {"node": "grade",
                                              "kept": len(update.get("documents", []))})
                    elif node == "rewrite_query":
                        yield _sse("status", {"node": "rewrite",
                                              "query": update.get("query_used", "")})
                    elif node == "web_search":
                        yield _sse("status", {"node": "web_search"})
                    elif node == "check_hallucination":
                        passed = update.get("hallucination_passed")
                        yield _sse("status", {"node": "check", "passed": passed})
                        if not passed:
                            # 本轮答案作废：连闸门一起丢弃（含正在扣留的尾巴），
                            # 下一轮 generate 会新建一个干净闸门，不会继承上一轮的 violated。
                            gate = None
                            yield _sse("reset", {})
            elif mode == "messages":
                # payload 形如 (AIMessageChunk, metadata)，metadata.langgraph_node 标明来源节点
                msg, meta = payload
                content = msg.content if isinstance(msg.content, str) else ""
                node = meta.get("langgraph_node")
                if node in _GATED_NODES and content:
                    # 走闸门：未校验完的数字扣在 buffer 里，本帧可能什么都不放行
                    if gate is None:
                        gate = NumericHoldBackGate(corpus_norm)
                    content = gate.feed(content)
                if node in _ANSWER_NODES and content:
                    sent_tokens += len(content)
                    yield _sse("token", {"t": content})

        # 流已结束：把闸门扣住的尾巴放行（否则以数字结尾的答案会永久少一截）
        if gate is not None:
            tail = gate.flush()
            if tail:
                sent_tokens += len(tail)
                yield _sse("token", {"t": tail})

        # 兜底：个别 OpenAI 兼容网关不支持流式回调 → 一次性补发完整答案
        if sent_tokens == 0:
            final_state = await asyncio.to_thread(
                engine.graph.get_state, config
            )
            answer_text = final_state.values.get("final_answer") or final_state.values.get("answer", "")
            for chunk in _split_tokens(answer_text):
                yield _sse("token", {"t": chunk})
        else:
            final_state = await asyncio.to_thread(engine.graph.get_state, config)
            answer_text = final_state.values.get("final_answer") or final_state.values.get("answer", "")

        cits = _citations(final_state.values)
        for piece in cits:
            yield _sse("citation", piece)
        elapsed = int((time.time() - t0) * 1000)
        yield _sse("done", {"answer": answer_text, "cached": False,
                            "elapsed_ms": elapsed})

        # 落历史（MySQL/SQLite）+ 缓存（只缓存通过数值校验的成功回答，fallback 话术不缓存）
        await asyncio.to_thread(store.append_turn,
                                thread_id, req.question, answer_text, elapsed, False)
        if fresh_session and answer_text and cits and final_state.values.get("hallucination_passed") is True:
            engine.cache_set(req.question, {"answer": answer_text, "citations": cits})

    except Exception as e:   # 任何环节炸了都以前端能渲染的方式收尾，绝不静默断流
        yield _sse("error", {"message": f"{type(e).__name__}: {e}"})


def _split_tokens(text: str, size: int = 6):
    """把整段答案切成小块逐帧发，模拟打字机节奏（供缓存命中 / 非流式兜底复用）。"""
    return [text[i:i + size] for i in range(0, len(text), size)]


# ---------------- 路由 ----------------

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式问答。事件类型：route / status / trace / token / citation / done / error。"""
    if not engine.ready:
        raise HTTPException(503, "引擎尚未完成冷启动（模型加载中），请稍后重试")
    return EventSourceResponse(_chat_gen(req))


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """上传/指定 PDF 触发解析入库并重建索引。"""
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(404, f"文件不存在: {path}")
    if path.suffix.lower() != ".pdf":
        raise HTTPException(400, "仅支持 PDF")
    try:
        n = await asyncio.to_thread(engine.ingest, path, req.company, req.year)
    except Exception as e:
        raise HTTPException(500, f"入库失败: {type(e).__name__}: {e}")
    return {"ok": True, "file": path.name, "chunks": n, "total_chunks": len(engine.retriever.docs)}


@app.post("/api/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)):
    """multipart 上传 PDF：保存到 data/raw 后走同一条入库链路。"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "仅支持 PDF")
    dest = settings.raw_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    return await ingest(IngestRequest(path=str(dest)))


@app.get("/api/documents")
async def documents():
    """已入库文档列表：来源文件、页数、分块数。"""
    stats: dict[str, dict] = {}
    for d in engine.retriever.docs:
        src = str(d.metadata.get("source", "unknown"))
        s = stats.setdefault(src, {"source": src, "pages": set(), "chunks": 0})
        s["pages"].add(d.metadata.get("page"))
        s["chunks"] += 1
    out = [{"source": v["source"], "pages": len(v["pages"]), "chunks": v["chunks"]}
           for v in stats.values()]
    return {"documents": out, "total_chunks": len(engine.retriever.docs)}


@app.get("/api/history/{session_id}")
async def history(session_id: str):
    """会话历史（MySQL/SQLite 持久化，跨重启可用）。"""
    turns = await asyncio.to_thread(store.load_history, session_id)
    return {"session_id": session_id, "turns": turns}


@app.get("/api/store/stats")
async def store_stats():
    """存储层运维视图：后端类型 / 总轮次 / 会话数 / 最近写入。"""
    return await asyncio.to_thread(store.stats)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        engine_ready=engine.ready,
        llm_provider=settings.llm_provider,
        doc_count=len(set(d.metadata.get("source") for d in engine.retriever.docs)) if engine.retriever else 0,
        chunk_count=len(engine.retriever.docs) if engine.retriever else 0,
    )
