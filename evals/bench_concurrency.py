"""
并发压测骨架（P1 agentic-rag-finance）——按路径分解的性能表。

## 为什么不用 locust

`locust` 默认按**响应头到达**计时。对 SSE 来说，响应头在第一个字节就到了，
于是流式请求会被记成「3 ms」——比真实快两个数量级，是假数据。
本项目必须**消费 SSE 直到 `done`** 才算一次请求完成，所以自写
`httpx.AsyncClient` + `asyncio.Semaphore`，天然支持流式读。

## 三条路径（对应规格 §1）

| 路径 | 打什么 | 测什么 |
|---|---|---|
| `a` 缓存命中 | `POST /api/chat`，已热身问题 + 全新 session | 应用层极限 + 落库锁 |
| `b` 冷路径 | `POST /api/chat`，未热身问题 + 全新 session | LLM 串行往返 |
| `c` 纯检索 | 直接调 `HybridRetriever.search()` | 组件开销与线程串行化 |

外加 `crosstalk`（并发串场）与 `ramp`（限流爬坡）。

## 三个必须写对的解析细节（写错就是 300 条全超时）

1. **分隔符是 `\\r\\n\\r\\n`，不是 `\\n\\n`。**
   `sse_starlette/sse.py:213` 的 `DEFAULT_SEPARATOR = "\\r\\n"`，而
   `src/api/app.py:230` 调 `EventSourceResponse(_chat_gen(req))` **没传 `sep=`**，
   于是每帧以 `\\r\\n\\r\\n` 结尾。`\\r\\n\\r\\n` 里**不存在** `\\n\\n` 子串
   （字节是 `\\r \\n \\r \\n`），所以用 `split("\\n\\n")` 会得到一坨永不切分的 buffer，
   一条 `done` 都读不到。本文件的 `P1_S_SEP` 只认 `\\r\\n\\r\\n`，
   并且**刻意不兼容 `\\n\\n`**——让「用错格式」在单测里就能被抓到。
2. **每 15s 有一条 `: ping - <时间>` 注释帧**（`DEFAULT_PING_INTERVAL=15`）。
   解析器必须跳过 `:` 开头的行，否则会把它当成坏帧。
3. **「流正常结束但没有 done」必须判失败。**
   `src/api/app.py:214-215` 的 `except` 只 `yield _sse("error", ...)`，
   **没有 finally、不补 done**。若按「HTTP 流结束 = 请求完成」计时，
   所有 error 请求都会被记成「极快的成功」，错误率 0%、P50 失真、整张表作废。

## 一个免费的隐藏指标：`delta = 客户端墙钟 - done.elapsed_ms`

`app.py:205` 先 `yield _sse("done", {... "elapsed_ms": elapsed})`，
`app.py:209` 才 `await asyncio.to_thread(store.append_turn, ...)`。
`EventSourceResponse` 必须等生成器 return 才能关流，而生成器 return 发生在
`append_turn` 之后 —— 所以**客户端看到流结束的时刻 = 落库跑完的时刻**。

于是 `delta` 就是「落库 + 网络 + SSE」的直接读数。`store.py:47` 有一把
**进程级 `threading.Lock`**（`store.py:89` 持有），并发一高就会排队。
只要对比「并发 1 的 delta」与「并发 N 的 delta」，就能实锤锁竞争，
**不需要给服务端加任何埋点**。这是本次压测最锐利的一把尺子。

## 语法

    # 0) 先自检解析器（不联网、不花钱）
    .venv/Scripts/python.exe -m pytest tests/test_bench_concurrency.py -q

    # 1) 量测管线冒烟：只发 1 条
    .venv/Scripts/python.exe evals/bench_concurrency.py --path a --dry-run

    # 2) A 路径（服务需已就绪：/health 的 engine_ready=true）
    .venv/Scripts/python.exe evals/bench_concurrency.py --path a --warm 6 -n 30 -c 1
    .venv/Scripts/python.exe evals/bench_concurrency.py --path a --skip-warm -n 300 -c 10

    # 3) B 路径（预热过的问题会被自动排除）
    .venv/Scripts/python.exe evals/bench_concurrency.py --path b -n 20 -c 2

    # 4) C 路径（脚本级，务必先停掉服务，否则两个进程抢 8GB 显存）
    .venv/Scripts/python.exe evals/bench_concurrency.py --path c --rerank off -n 100 -t 4

    # 5) 串场 + 爬坡
    .venv/Scripts/python.exe evals/bench_concurrency.py --path crosstalk -c 6
    .venv/Scripts/python.exe evals/bench_concurrency.py --path ramp --levels 2 5 10 20

问题分组是**确定性**的：`--split warm` 取 goldset 前 6 条，`--split cold` 取后 20 条，
两者恒不重叠（`cache_key` 带 3600s TTL，重叠会让冷路径样本作废）。

输出：终端 markdown 行（可直接贴进 METRICS）+ `evals/concurrency_<path>.json`。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDSET = ROOT / "evals" / "goldset.json"

# sse_starlette 的默认帧分隔符。只认这个，不兼容 "\n\n"（见模块 docstring 第 1 条）。
P1_S_SEP = b"\r\n\r\n"

# goldset 26 条怎么分：6 热身 + 20 冷路径 = 26，正好不重叠。
N_WARM = 6
N_COLD = 20


# ============================================================================
# SSE 解析
# ============================================================================
def parse_p1_frame(raw: bytes) -> dict:
    """把一帧（不含结尾分隔符）拆成 {event, data, comment}。

    不做 JSON 解码，只切字段——解码留给调用方，便于坏帧计数而不是整体崩。
    """
    event = None
    data_lines: list[str] = []
    comments: list[str] = []
    for line in raw.decode("utf-8", "replace").split("\n"):
        line = line.rstrip("\r")
        if not line:
            continue
        if line.startswith(":"):
            comments.append(line[1:].strip())
            continue
        field, sep, value = line.partition(":")
        if not sep:
            continue
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    return {
        "event": event,
        "data": "\n".join(data_lines) if data_lines else None,
        "comment": comments[0] if (comments and not data_lines) else None,
    }


class P1SseParser:
    """增量解析 P1 的 SSE 流。

    状态机刻意保持「只进不退」：`done` 一旦被置上就不再改动，`error` 同理。
    这样「既没 done 也没 error」可以直接判成断流，不需要额外的标志位。
    """

    def __init__(self) -> None:
        self._buf = b""
        self.n_frames = 0          # 有效数据帧（含 ping 注释以外的帧）
        self.n_pings = 0
        self.n_bad = 0             # JSON 解不开 / 缺 data 的帧
        self.n_tokens = 0
        self.done: dict | None = None
        self.error: str | None = None
        self.route: str | None = None

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._buf += chunk
        while True:
            idx = self._buf.find(P1_S_SEP)
            if idx < 0:
                break
            raw, self._buf = self._buf[:idx], self._buf[idx + len(P1_S_SEP):]
            self._handle(parse_p1_frame(raw))

    def _handle(self, frame: dict) -> None:
        if frame["comment"] is not None:
            self.n_pings += 1
            return
        payload_raw = frame["data"]
        if not payload_raw:
            self.n_bad += 1
            return
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            self.n_bad += 1
            return
        self.n_frames += 1
        ev = frame["event"]
        if ev == "token":
            self.n_tokens += 1
        elif ev == "route":
            self.route = payload.get("route")
        elif ev == "done":
            self.done = payload
        elif ev == "error":
            self.error = payload.get("message") or "error"


def classify_outcome(parser: P1SseParser) -> tuple[bool, str, str | None]:
    """把解析器终态归成 (ok, outcome, error)。

    单独抽成纯函数是为了能被单测直接钉住——**「流结束但没 done」必须判失败**，
    这条如果写在 `chat_once` 里就只能靠起服务才测得到，而它恰恰是最容易写错、
    写错了又最不容易发现的一条（错误请求会被记成「极快的成功」）。
    """
    if parser.error is not None:
        return False, "sse_error", parser.error
    if parser.done is None:
        return False, "eof_no_done", f"流结束但未收到 done（有效帧 {parser.n_frames}）"
    return True, "done", None


# ============================================================================
# 指标
# ============================================================================
def percentile(values: list[float], p: float) -> float | None:
    """最近秩法（nearest-rank）。样本少时比插值法更保守，也不会有小数级幻读。"""
    if not values:
        return None
    s = sorted(values)
    k = int(math.ceil(p / 100.0 * len(s))) - 1
    return s[min(max(k, 0), len(s) - 1)]


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def summarize(records: list[dict], wall_s: float) -> dict:
    n = len(records)
    ok = [r for r in records if r["ok"]]
    wall = [r["wall_ms"] for r in ok]
    elapsed = [r["elapsed_ms"] for r in ok if r.get("elapsed_ms") is not None]
    delta = [r["delta_ms"] for r in ok if r.get("delta_ms") is not None]
    out = {
        "n": n,
        "n_ok": len(ok),
        "n_fail": n - len(ok),
        "qps": _r(len(ok) / wall_s, 3) if wall_s > 0 else None,
        "wall_s": _r(wall_s, 2),
        "p50_ms": _r(percentile(wall, 50)),
        "p95_ms": _r(percentile(wall, 95)),
        "p99_ms": _r(percentile(wall, 99)),
        "mean_ms": _r(sum(wall) / len(wall)) if wall else None,
        "err_rate": _r((n - len(ok)) / n, 4) if n else None,
        "server_elapsed_p50_ms": _r(percentile(elapsed, 50)),
        "server_elapsed_p95_ms": _r(percentile(elapsed, 95)),
        "delta_p50_ms": _r(percentile(delta, 50)),
        "delta_p95_ms": _r(percentile(delta, 95)),
        "outcomes": {},
    }
    for r in records:
        out["outcomes"][r["outcome"]] = out["outcomes"].get(r["outcome"], 0) + 1
    return out


def md_row(name: str, n: int, conc, s: dict, bottleneck: str) -> str:
    return ("| {name} | {n} | {c} | {qps} | {p50} | {p95} | {p99} | {err} | {bn} |").format(
        name=name, n=n, c=conc,
        qps=f"{s['qps']:.2f}" if s["qps"] is not None else "-",
        p50=f"{s['p50_ms']:.0f}ms" if s["p50_ms"] is not None else "-",
        p95=f"{s['p95_ms']:.0f}ms" if s["p95_ms"] is not None else "-",
        p99=f"{s['p99_ms']:.0f}ms" if s["p99_ms"] is not None else "-",
        err=f"{s['err_rate'] * 100:.1f}%" if s["err_rate"] is not None else "-",
        bn=bottleneck,
    )


# ============================================================================
# goldset 分组（确定性，避免热身与冷路径重叠）
# ============================================================================
def load_items() -> list[dict]:
    data = json.loads(GOLDSET.read_text(encoding="utf-8"))
    return data["items"]


def pick_items(split: str) -> list[dict]:
    items = load_items()
    if split == "warm":
        return items[:N_WARM]
    if split == "cold":
        return items[N_WARM:]
    if split == "all":
        return items
    raise SystemExit(f"未知 split: {split}")


# ============================================================================
# HTTP 打流
# ============================================================================
async def chat_once(client, url: str, item: dict, session_id: str, sem) -> dict:
    """发一条 /api/chat，消费 SSE 直到 done / error / EOF。"""
    t0 = time.perf_counter()
    rec = {
        "qid": item.get("id"),
        "question": item["question"],
        "session_id": session_id,
        "ok": False,
        "outcome": "unset",
        "http_status": None,
        "wall_ms": None,
        "elapsed_ms": None,
        "delta_ms": None,
        "cached": None,
        "route": None,
        "n_frames": 0,
        "n_tokens": 0,
        "answer": "",
        "error": None,
    }
    parser = P1SseParser()
    async with sem:
        try:
            async with client.stream(
                "POST", url, json={"question": item["question"], "session_id": session_id}
            ) as resp:
                rec["http_status"] = resp.status_code
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    rec["outcome"] = "http_error"
                    rec["error"] = f"HTTP {resp.status_code}: {body}"
                else:
                    # aiter_raw 而非 aiter_lines：后者会吃掉行尾，对 \r\n 语义模糊
                    async for chunk in resp.aiter_raw():
                        parser.feed(chunk)
                    rec["n_frames"] = parser.n_frames
                    rec["n_tokens"] = parser.n_tokens
                    rec["route"] = parser.route
                    ok, outcome, err = classify_outcome(parser)
                    rec["ok"] = ok
                    rec["outcome"] = outcome
                    rec["error"] = err
                    if ok:
                        rec["answer"] = parser.done.get("answer") or ""
                        rec["cached"] = parser.done.get("cached")
                        rec["elapsed_ms"] = parser.done.get("elapsed_ms")
        except Exception as e:                     # noqa: BLE001 —— 超时/断连都要留记录
            rec["outcome"] = "exception"
            rec["error"] = f"{type(e).__name__}: {e}"

    rec["wall_ms"] = (time.perf_counter() - t0) * 1000.0
    if rec["elapsed_ms"] is not None:
        rec["delta_ms"] = rec["wall_ms"] - rec["elapsed_ms"]
    if rec["ok"]:
        rec["wall_ms"] = _r(rec["wall_ms"])
        rec["delta_ms"] = _r(rec["delta_ms"])
    else:
        rec["wall_ms"] = _r(rec["wall_ms"])
    return rec


async def run_http(url: str, items: list[dict], n: int, conc: int,
                   timeout_s: float, fresh_session: bool = True,
                   progress: bool = True) -> tuple[list[dict], float]:
    """按并发上限打 n 条。返回 (records, 墙钟秒)。

    session 语义：`fresh_session=True` 时每条独立 uuid —— 这是缓存路径的**硬前提**
    （`fresh_session = not await store.load_history(thread_id)`，见 `app.py:93-95`），
    共用 session 会让缓存永远不命中。
    """
    import httpx

    limits = httpx.Limits(max_connections=conc + 8, max_keepalive_connections=conc + 8)
    timeout = httpx.Timeout(timeout_s, connect=10.0)
    sem = asyncio.Semaphore(conc)

    async def worker(item: dict, sid: str) -> dict:
        """每条请求自建 client。

        共用一个 client 时，连接池上限（`max_connections`）会变成隐式限流器，
        把「服务端慢」与「客户端排不上连接」混在一起，读数就没有归因价值了。
        300 条规模下多开 socket 的开销可以忽略。
        """
        async with httpx.AsyncClient(
            timeout=timeout, limits=limits, follow_redirects=False
        ) as cli:
            return await chat_once(cli, url, item, sid, sem)

    tasks = []
    for i in range(n):
        item = items[i % len(items)]
        sid = f"lt-{uuid.uuid4().hex}" if fresh_session else f"lt-fixed-{i}"
        tasks.append(asyncio.create_task(worker(item, sid)))

    t0 = time.perf_counter()
    records: list[dict] = []
    for fut in asyncio.as_completed(tasks):
        records.append(await fut)
        if progress and len(records) % 20 == 0:
            print(f"  ... {len(records)}/{n}", flush=True)
    return records, time.perf_counter() - t0


# ============================================================================
# 各路径
# ============================================================================
def _save(path_key: str, tag: str, payload: dict) -> Path:
    out = ROOT / "evals" / f"concurrency_{path_key}{('_' + tag) if tag else ''}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out.relative_to(ROOT).as_posix()}")
    return out


def cmd_cache(a, url: str) -> int:
    """A 路径：热身 -> 验缓存 -> 打流量。

    热身闸门不能只看 HTTP 200：`cache_set` 要同时满足
    `fresh_session` / `answer_text` 非空 / `citations` 非空 / `hallucination_passed`
    四个条件（`app.py:211-212`），而第四条在响应里看不见。
    唯一可靠的验证是**用新 session 再问一次，断言 cached==true**。
    """
    warm_items = pick_items("warm")
    items = warm_items[:1] if a.questions == 1 else warm_items[:a.questions]

    if not a.skip_warm:
        print(f"[热身] {len(warm_items)} 条问题，各跑一次完整图（走 LLM）...")
        for it in warm_items:
            rec = asyncio.run(_single(url, it, f"lt-warm-{uuid.uuid4().hex}", a.timeout))
            flag = "done" if rec["ok"] else rec["outcome"]
            print(f"  {it['id']}: {flag}  cached={rec['cached']}  "
                  f"{rec['wall_ms']:.0f}ms  err={rec['error']}")

    print("[验缓存] 每个热身问题再用新 session 问一次，必须 cached=true ...")
    verify = []
    for it in warm_items:
        rec = asyncio.run(_single(url, it, f"lt-verify-{uuid.uuid4().hex}", a.timeout))
        verify.append(rec)
        print(f"  {it['id']}: cached={rec['cached']}  outcome={rec['outcome']}")
    n_hit = sum(1 for r in verify if r["cached"] is True)
    if n_hit < len(warm_items):
        print(f"[FAIL] 缓存闸门未过：{n_hit}/{len(warm_items)} 命中。"
              f"先查 session 是否唯一、问题字符串是否与热身逐字节一致（标点差异致命）、"
              f"服务是否被重启过（fakeredis 随进程消失）。")
        return 2

    print(f"\n[打流量] {a.n} 条 @ 并发 {a.conc}，问题复用 '{items[0]['id']}'，session 全新建")
    records, wall_s = asyncio.run(
        run_http(url, items, a.dry_run and 1 or a.n, a.conc, a.timeout)
    )
    s = summarize(records, wall_s)
    _print_summary(s, records)
    hit_rate = sum(1 for r in records if r["cached"] is True) / max(len(records), 1)
    print(f"cached=true 占比：{hit_rate * 100:.1f}%  "
          f"（必须 100%，否则该档数字作废）")
    print(f"A 路径  -> {md_row('P1-A 缓存命中', len(records), a.conc, s, _guess(records, s))}")
    _save("A", a.tag, {"meta": _meta(a, url), "summary": s, "records": records})
    return 0 if s["n_fail"] == 0 and hit_rate == 1.0 else 1


def cmd_cold(a, url: str) -> int:
    """B 路径：20 条未热身问题，全新 session。"""
    items = pick_items("cold")
    n = a.dry_run and 1 or min(a.n, len(items))
    print(f"[冷路径] {n} 条问题（{items[0]['id']} .. {items[-1]['id']}），session 全新建")
    print("           这些 id 与 --split warm 的前 6 条恒不重叠")
    records, wall_s = asyncio.run(run_http(url, items, n, a.conc, a.timeout))
    s = summarize(records, wall_s)
    _print_summary(s, records)
    cached = sum(1 for r in records if r["cached"] is True)
    if cached:
        print(f"[WARN] {cached} 条返回 cached=true —— 撞上了 A 路径的缓存，这些样本应剔除")
    acc = _answer_accuracy(records)
    print(f"answer_substr 命中率：{acc * 100:.1f}%（串行基线 22/26 = 84.6%）")
    print(f"B 路径  -> {md_row('P1-B 冷路径', len(records), a.conc, s, _guess(records, s))}")
    _save("B", a.tag, {"meta": _meta(a, url), "summary": s,
                       "answer_accuracy": acc, "records": records})
    return 0 if s["err_rate"] is not None and s["err_rate"] <= 0.05 else 1


def cmd_retrieval(a) -> int:
    """C 路径：直接调检索器，不起服务。

    **必须先停掉 API 服务**：BGE-M3 约 2.3GB + reranker 约 1.1GB，
    4070 Laptop 只有 8GB，两个进程会把显存打爆。
    """
    from concurrent.futures import ThreadPoolExecutor

    from src.api.engine import Engine
    from src.retrieval.hybrid import HybridRetriever

    use_rerank = a.rerank == "on"
    print(f"[C 路径] rerank={a.rerank}  threads={a.threads}  n={a.n}")

    t = time.perf_counter()
    eng = Engine()
    docs, _ = eng.build_corpus()
    print(f"语料加载：{len(docs):,} 块，{time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    retr = HybridRetriever(docs, use_rerank=use_rerank)
    print(f"检索器构建：{time.perf_counter() - t:.2f}s  "
          f"n_reencoded={getattr(retr, 'n_reencoded', '?')}")
    if use_rerank and retr.reranker is None:
        print("[FAIL] 要求 rerank=on 但 reranker 为 None —— 开关没生效，"
              "此时得出的「精排无用」结论是假的")
        return 2
    if not use_rerank and retr._use_rerank:
        print("[FAIL] 要求 rerank=off 但 _use_rerank 仍为 True")
        return 2

    items = pick_items("all")
    queries = [items[i % len(items)]["question"] for i in range(a.n)]

    def one(q: str) -> float:
        t0 = time.perf_counter()
        hits = retr.search(q, k=5)
        dt = (time.perf_counter() - t0) * 1000.0
        # 决定性断言：两种模式都必须返回 (Document, score)
        assert isinstance(hits[0], tuple) and len(hits[0]) == 2, \
            f"search() 必须返回 (Document, score)，实际 {type(hits[0])}"
        return dt

    # 先串行跑一遍做基线，再按 --threads 并发
    t0 = time.perf_counter()
    serial = [one(q) for q in queries[: min(10, a.n)]]
    serial_s = time.perf_counter() - t0
    print(f"串行基线（{len(serial)} 条）：P50 {percentile(serial, 50):.1f}ms  "
          f"吞吐 {len(serial) / serial_s:.2f} q/s")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.threads) as pool:
        par = list(pool.map(one, queries))
    par_s = time.perf_counter() - t0
    s = {
        "n": len(par), "n_ok": len(par), "n_fail": 0,
        "qps": _r(len(par) / par_s, 3), "wall_s": _r(par_s, 2),
        "p50_ms": _r(percentile(par, 50)), "p95_ms": _r(percentile(par, 95)),
        "p99_ms": _r(percentile(par, 99)), "mean_ms": _r(sum(par) / len(par)),
        "err_rate": 0.0, "outcomes": {"done": len(par)},
        "server_elapsed_p50_ms": None, "server_elapsed_p95_ms": None,
        "delta_p50_ms": None, "delta_p95_ms": None,
    }
    print(f"并发 {a.threads}：吞吐 {s['qps']:.2f} q/s  "
          f"P50 {s['p50_ms']:.1f}ms  P95 {s['p95_ms']:.1f}ms")
    ratio = (s["qps"] / (len(serial) / serial_s)) if serial_s > 0 else None
    print(f"并发/串行 吞吐比：{ratio:.2f}x  "
          f"（若约等于 1 而不是 {a.threads}x，就是 CUDA 串行化 + GIL，"
          f"这本身就是结论，不要修）")
    label = f"P1-C 纯检索(rerank={a.rerank})"
    print(f"C 路径  -> {md_row(label, len(par), a.threads, s, 'CUDA/GIL 串行化' if ratio and ratio < 1.5 * 1 else '见 throughput ratio')}")
    _save("C_" + a.rerank, a.tag, {
        "meta": _meta(a, None) | {"use_rerank": use_rerank,
                                  "n_reencoded": getattr(retr, "n_reencoded", None),
                                  "serial_p50_ms": _r(percentile(serial, 50)),
                                  "throughput_ratio": _r(ratio)},
        "summary": s, "serial_ms": [_r(x) for x in serial],
        "parallel_ms": [_r(x) for x in par],
    })
    return 0


def cmd_crosstalk(a, url: str) -> int:
    """并发正确性：N 个不同问题同时发，检查答案有没有串场。

    `answer_substr` 都是 `170,899,152,276.34` 这种高熵数字串，交叉污染极易检出。
    **任何一条命中别人的 answer_substr 就是阻断级缺陷**，先修再谈性能。
    """
    items = pick_items("warm")[:a.questions]
    print(f"[串场] {len(items)} 个不同问题 @ 并发 {len(items)}")

    # 先确保都被缓存（否则每条都要走完整图，既慢又贵）
    for it in items:
        asyncio.run(_single(url, it, f"lt-warm-{uuid.uuid4().hex}", a.timeout))

    records, wall_s = asyncio.run(
        run_http(url, items, len(items), len(items), a.timeout)
    )
    own = 0
    foreign = []
    for rec in records:
        ans = rec.get("answer") or ""
        me = next((it for it in items if it["id"] == rec["qid"]), None)
        if me and me["answer_substr"] and me["answer_substr"] in ans:
            own += 1
        for other in items:
            if other["id"] == rec["qid"]:
                continue
            if other["answer_substr"] and other["answer_substr"] in ans:
                foreign.append((rec["qid"], other["id"]))

    s = summarize(records, wall_s)
    _print_summary(s, records)
    print(f"命中自己金标：{own}/{len(records)}")
    print(f"命中他人金标（串场）：{len(foreign)} 处"
          + (f"  -> {foreign[:5]}" if foreign else "  [OK] 0 串场"))
    _save("crosstalk", a.tag, {"meta": _meta(a, url), "summary": s,
                               "own_hit": own, "foreign_hits": foreign,
                               "records": records})
    return 0 if not foreign else 3


def cmd_ramp(a, url: str) -> int:
    """限流爬坡：并发 2 -> 5 -> 10 -> 20，看从哪一档开始崩。

    **只打缓存命中路径**。冷路径爬 2->10 要 200 次全 RAG（约 1 小时、约 6 毛钱），
    而瓶颈早已确定是 LLM 串行往返 —— 爬坡要回答的是「在哪崩」，只有便宜路径爬得到崩点。
    """
    items = pick_items("warm")[:1]
    if not a.skip_warm:
        print("[热身] 1 条问题...")
        asyncio.run(_single(url, items[0], f"lt-warm-{uuid.uuid4().hex}", a.timeout))

    rows = []
    print()
    print("| 并发 | 请求 | QPS | P50 | P95 | P99 | 错误率 | 主要收尾 |")
    print("|---|---|---|---|---|---|---|---|")
    for conc in a.levels:
        records, wall_s = asyncio.run(
            run_http(url, items, a.n, conc, a.timeout, progress=False)
        )
        s = summarize(records, wall_s)
        top = max(s["outcomes"].items(), key=lambda kv: kv[1])
        rows.append({"concurrency": conc, "summary": s})
        print(f"| {conc} | {len(records)} | {s['qps']:.2f} | {s['p50_ms']:.0f}ms | "
              f"{s['p95_ms']:.0f}ms | {s['p99_ms']:.0f}ms | "
              f"{s['err_rate'] * 100:.1f}% | {top[0]} |")
        if s["delta_p95_ms"] is not None and s["delta_p50_ms"] is not None:
            print(f"|     | delta P50/P95 | {s['delta_p50_ms']:.1f} / "
                  f"{s['delta_p95_ms']:.1f} ms | | | | |")
    _save("ramp", a.tag, {"meta": _meta(a, url), "rows": rows})
    return 0


# ============================================================================
# 辅助
# ============================================================================
async def _single(url: str, item: dict, session_id: str, timeout_s: float) -> dict:
    import httpx
    sem = asyncio.Semaphore(1)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s, connect=10.0), follow_redirects=False
    ) as cli:
        return await chat_once(cli, url, item, session_id, sem)


def _answer_accuracy(records: list[dict]) -> float:
    items = {it["id"]: it for it in load_items()}
    hit = 0
    for rec in records:
        it = items.get(rec.get("qid"))
        if it and it.get("answer_substr") and it["answer_substr"] in (rec.get("answer") or ""):
            hit += 1
    return hit / len(records) if records else 0.0


def _guess(records: list[dict], s: dict) -> str:
    """给这张表补一句**可被证伪**的瓶颈猜测，而不是空着等事后补。"""
    if s["p50_ms"] is None:
        return "全失败"
    if s["p95_ms"] and s["p50_ms"] and s["p95_ms"] > 3 * s["p50_ms"]:
        return "长尾：疑似落库锁排队"
    if (s["delta_p95_ms"] or 0) > 50:
        return "落库（store.py:_lock）"
    return "待定（看 delta 与帧数）"


def _meta(a, url) -> dict:
    meta = {
        "path": getattr(a, "path", None),
        "n": getattr(a, "n", None),
        "concurrency": getattr(a, "conc", None),
        "timeout_s": getattr(a, "timeout", None),
        "url": url,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": getattr(a, "tag", ""),
    }
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:                              # noqa: BLE001
        meta["torch"] = None
    return meta


def _print_summary(s: dict, records: list[dict]) -> None:
    print()
    print(f"完成 {s['n_ok']}/{s['n']}  QPS {s['qps']}  "
          f"P50 {s['p50_ms']}ms  P95 {s['p95_ms']}ms  P99 {s['p99_ms']}ms  "
          f"错误率 {s['err_rate'] * 100:.1f}%")
    print(f"收尾分布：{s['outcomes']}")
    if s.get("delta_p50_ms") is not None:
        print(f"服务端 elapsed P50 {s['server_elapsed_p50_ms']}ms vs "
              f"客户端墙钟 P50 {s['p50_ms']}ms -> delta P50 {s['delta_p50_ms']}ms "
              f"P95 {s['delta_p95_ms']}ms")
    bad = [r for r in records if not r["ok"]]
    for r in bad[:5]:
        print(f"  [FAIL] {r['qid']} {r['outcome']}: {r['error']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="P1 并发压测（SSE 消费到 done 才计时）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--path", required=True,
                   choices=["a", "b", "c", "crosstalk", "ramp"],
                   help="a=缓存命中 b=冷路径 c=纯检索 crosstalk=串场 ramp=爬坡")
    p.add_argument("--base-url", default="http://127.0.0.1:8300")
    p.add_argument("-n", "--n", type=int, default=0, help="请求数（0=用各路径默认值）")
    p.add_argument("-c", "--concurrency", type=int, default=0, dest="conc")
    p.add_argument("--questions", type=int, default=1,
                   help="用几条不同问题（a/crosstalk 用）")
    p.add_argument("--warm", type=int, default=N_WARM, help="（保留）热几条")
    p.add_argument("--skip-warm", action="store_true", help="跳过热身，只验缓存")
    p.add_argument("--rerank", choices=["on", "off"], default="off", help="C 路径用")
    p.add_argument("-t", "--threads", type=int, default=4, help="C 路径线程数")
    p.add_argument("--levels", type=int, nargs="+", default=[2, 5, 10, 20],
                   help="ramp 的并发档位")
    p.add_argument("--timeout", type=float, default=180.0,
                   help="单请求超时秒（冷路径最坏 60s，留 3 倍余量）")
    p.add_argument("--tag", default="", help="结果文件后缀，用于区分模型/环境")
    p.add_argument("--dry-run", action="store_true", help="只发 1 条，验证管线")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    url = a.base_url.rstrip("/") + "/api/chat"

    # 每路径的默认值集中在 main 里，避免 argparse 的 default 互相干扰
    defaults = {"a": (300, 10), "b": (20, 2), "c": (100, 4), "crosstalk": (6, 6),
                "ramp": (30, 2)}
    dn, dc = defaults[a.path]
    if a.n <= 0:
        a.n = dn
    if a.conc <= 0:
        a.conc = dc
    if a.path == "crosstalk":
        a.questions = max(a.questions, 3)
        a.conc = a.questions

    if a.path == "c":
        return cmd_retrieval(a)
    if not _health_ok(a.base_url, a.path in ("a", "ramp", "crosstalk")):
        return 4
    if a.path == "a":
        return cmd_cache(a, url)
    if a.path == "b":
        return cmd_cold(a, url)
    if a.path == "crosstalk":
        return cmd_crosstalk(a, url)
    return cmd_ramp(a, url)


def _health_ok(base_url: str, need_engine: bool) -> bool:
    """就绪闸门。`/api/chat` 在 engine 未就绪时直接抛 503（app.py:228-229），
    不等就绪就开打，会把 503 记成业务错误率。"""
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=10) as r:
            body = json.loads(r.read().decode("utf-8"))
    except Exception as e:                         # noqa: BLE001
        print(f"[FAIL] /health 不可达：{type(e).__name__}: {e}")
        print("       先起服务：.venv/Scripts/python.exe scripts/run_api.py")
        return False
    ready = body.get("engine_ready")
    print(f"/health engine_ready={ready}")
    if need_engine and not ready:
        print("[FAIL] 引擎尚未完成冷启动（GPU 约 15s / CPU 约 33s），稍后再试")
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
