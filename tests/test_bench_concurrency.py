"""
压测脚本的解析器自证（P1）。

## 为什么这一组用例比压测数字更重要

压测代码属于「计量器」。计量器错了，跑出来的每个数字都是错的，而且**看起来很正常**——
P50 有值、错误率 0%、表格漂亮。所以先钉住计量器，再去测系统。

本文件钉的都是「错了也不会报错、只会静默给出错误数字」的地方：

1. **分隔符**：P1 是 `\\r\\n\\r\\n`（`sse_starlette/sse.py:213` 的默认值），不是 `\\n\\n`。
   用 `\\n\\n` 切会得到一坨永不切分的 buffer → 一条 `done` 都读不到 → 300 条全超时。
2. **跨包**：一个帧可能被 TCP 切在分隔符中间。
3. **心跳**：`sse_starlette` 每 15s 发 `: ping - <时间>`，不能当坏帧。
4. **EOF 无 done**：`app.py:214-215` 的 except 不补 `done`。
   若按「流结束 = 完成」，error 请求会被记成「极快的成功」。

第 5 条 `test_p1_parser_rejects_p2_frames` 是**决定性断言**：只断言「能解析 P1 帧」不够——
两个来源给冲突值才算钉住。喂 P2 的 `data:`-only 帧进去，必须解析不出 `done`。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_bench():
    """从路径加载 evals/bench_concurrency.py（evals 不是包，不能直接 import）。"""
    path = ROOT / "evals" / "bench_concurrency.py"
    spec = importlib.util.spec_from_file_location("bench_concurrency_p1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()


# P1 的真实帧形态：`event: X` + `data: {...}`，以 \r\n\r\n 分隔（sse_starlette 默认）
def _p1_stream(frames: list[tuple[str, dict]]) -> bytes:
    out = b""
    for event, payload in frames:
        out += f"event: {event}\r\ndata: {json.dumps(payload, ensure_ascii=False)}\r\n\r\n".encode()
    return out


def _p2_stream(events: list[dict]) -> bytes:
    return b"".join(
        f"data: {json.dumps(e, ensure_ascii=False)}\n\n".encode() for e in events
    )


# ---------------------------------------------------------------- 帧解析
def test_parse_p1_frame_reads_event_and_data():
    frame = bench.parse_p1_frame(b'event: done\r\ndata: {"answer": "x", "cached": true}')
    assert frame["event"] == "done"
    assert json.loads(frame["data"]) == {"answer": "x", "cached": True}
    assert frame["comment"] is None


def test_parse_p1_frame_treats_colon_line_as_comment():
    """`:` 开头是 SSE 注释帧（心跳就走这条路），必须和 `data:` 区分开。"""
    frame = bench.parse_p1_frame(b": ping - 2026-10-20 16:00:00+00:00")
    assert frame["data"] is None
    assert frame["comment"].startswith("ping")


def test_parse_p1_frame_multiline_data_is_joined_by_newline():
    """SSE 规范允许多行 `data:`，按规范用 \\n 拼接。"""
    frame = bench.parse_p1_frame(b"event: token\r\ndata: {\"t\":\r\ndata: \"hi\"}")
    assert frame["event"] == "token"
    assert json.loads(frame["data"]) == {"t": "hi"}


# ---------------------------------------------------------------- 流式解析
def test_parser_walks_full_stream_to_done():
    stream = _p1_stream([
        ("route", {"route": "cache", "query": "q"}),
        ("citation", {"source": "a.pdf", "page": 5}),
        ("token", {"t": "170,899"}),
        ("token", {"t": "152,276.34"}),
        ("done", {"answer": "170,899,152,276.34", "cached": True, "elapsed_ms": 7}),
    ])
    p = bench.P1SseParser()
    p.feed(stream)
    assert p.done == {"answer": "170,899,152,276.34", "cached": True, "elapsed_ms": 7}
    assert p.n_frames == 5
    assert p.n_tokens == 2
    assert p.route == "cache"
    ok, outcome, err = bench.classify_outcome(p)
    assert ok is True and outcome == "done" and err is None


def test_parser_survives_frame_split_across_chunks():
    """一帧被切成任意位置的多块（含切在 \\r\\n\\r\\n 中间）都必须能拼回来。

    TCP 不保证帧边界，这个用例就是在模拟真实网络。
    """
    stream = _p1_stream([
        ("token", {"t": "a"}),
        ("done", {"answer": "abc", "cached": False, "elapsed_ms": 3}),
    ])
    p = bench.P1SseParser()
    for i in range(len(stream)):          # 逐字节喂，最坏情况
        p.feed(stream[i:i + 1])
    assert p.done is not None
    assert p.done["answer"] == "abc"
    assert p.n_tokens == 1


def test_parser_ignores_ping_comment_frames():
    """心跳帧不能算坏帧，也不能推进 n_frames。"""
    stream = (
        _p1_stream([("token", {"t": "a"})])
        + b": ping - 2026-10-20T16:00:00Z\r\n\r\n"
        + _p1_stream([("done", {"answer": "a", "cached": False, "elapsed_ms": 1})])
    )
    p = bench.P1SseParser()
    p.feed(stream)
    assert p.n_pings == 1
    assert p.n_frames == 2
    assert p.n_bad == 0
    assert p.done is not None


def test_parser_counts_bad_json_without_dying():
    """坏帧只计数不抛——一条脏帧不该毁掉整轮 300 条的统计。"""
    p = bench.P1SseParser()
    p.feed(b"event: token\r\ndata: {not json\r\n\r\n")
    p.feed(b"event: done\r\ndata: {\"answer\": \"ok\", \"cached\": false}\r\n\r\n")
    assert p.n_bad == 1
    assert p.done is not None


def test_p1_parser_rejects_p2_frames():
    """**决定性断言**：喂 P2 的 `data:`-only 帧，必须解析不出 `done`。

    只断言「能解析 P1 帧」是单边的——万一解析器实际上两种格式都认，
    这条就白测了。两个来源给冲突值才算钉住。
    """
    p2_bytes = _p2_stream([
        {"type": "thought", "step": "x"},
        {"type": "done", "history_id": "h1", "status": "ok"},
    ])
    p = bench.P1SseParser()
    p.feed(p2_bytes)
    assert p.done is None
    assert p.n_frames == 0            # 压根没切出帧来
    ok, outcome, _ = bench.classify_outcome(p)
    assert ok is False and outcome == "eof_no_done"


# ---------------------------------------------------------------- EOF 判定
def test_eof_without_done_is_not_success():
    """`app.py:214-215` 的 except 只发 error 不补 done。

    若按「流结束 = 完成」判定，这些请求会被记成「极快的成功」，
    于是错误率 0%、P50 失真——整张表作废。这条用例把判据钉死。
    """
    p = bench.P1SseParser()
    p.feed(_p1_stream([("route", {"route": "full"}), ("token", {"t": "半"})]))
    ok, outcome, err = bench.classify_outcome(p)
    assert ok is False
    assert outcome == "eof_no_done"
    assert "done" in err


def test_sse_error_frame_is_failure_even_if_stream_closes_cleanly():
    p = bench.P1SseParser()
    p.feed(_p1_stream([("error", {"message": "OperationalError: database is locked"})]))
    ok, outcome, err = bench.classify_outcome(p)
    assert ok is False
    assert outcome == "sse_error"
    assert "database is locked" in err


def test_error_then_done_still_reports_error():
    """error 优先于 done：一旦报过错，就不能被后面的 done 洗白。"""
    p = bench.P1SseParser()
    p.feed(_p1_stream([
        ("error", {"message": "boom"}),
        ("done", {"answer": "x", "cached": False}),
    ]))
    ok, outcome, _ = bench.classify_outcome(p)
    assert ok is False and outcome == "sse_error"


# ---------------------------------------------------------------- 指标
def test_percentile_uses_nearest_rank():
    vals = [float(i) for i in range(1, 101)]      # 1..100
    assert bench.percentile(vals, 50) == 50
    assert bench.percentile(vals, 95) == 95
    assert bench.percentile(vals, 99) == 99
    assert bench.percentile(vals, 100) == 100
    assert bench.percentile([], 50) is None


def test_percentile_on_small_sample_does_not_overrun():
    """样本少于 100 时，最近秩法容易越界——这里钉住不 IndexError。"""
    assert bench.percentile([7.0], 99) == 7.0
    assert bench.percentile([1.0, 2.0], 99) == 2.0


def test_summarize_counts_failures_and_delta():
    records = [
        {"ok": True, "wall_ms": 10.0, "elapsed_ms": 6, "delta_ms": 4.0,
         "outcome": "done"},
        {"ok": True, "wall_ms": 20.0, "elapsed_ms": 6, "delta_ms": 14.0,
         "outcome": "done"},
        {"ok": False, "wall_ms": 5.0, "elapsed_ms": None, "delta_ms": None,
         "outcome": "eof_no_done"},
    ]
    s = bench.summarize(records, wall_s=2.0)
    assert s["n"] == 3 and s["n_ok"] == 2 and s["n_fail"] == 1
    assert s["qps"] == 1.0
    assert s["err_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["delta_p50_ms"] == 4.0
    assert s["outcomes"] == {"done": 2, "eof_no_done": 1}


# ---------------------------------------------------------------- 问题配额
def test_warm_and_cold_splits_are_disjoint_and_cover_goldset():
    """热身与冷路径的问题**必须不重叠**。

    `cache_key` 带 3600s TTL，若冷路径用了热身过的问题，那条样本会返回
    `cached=true` —— 它测的就不再是冷路径。而 goldset 只有 26 条，
    规格要的「10 条热身 + 20 条冷路径」在数学上就不成立（30 > 26）。
    这条用例把 6/20 这个配额的**不重叠性**钉住，防止以后有人改 N_WARM
    把两个集合撞上。
    """
    warm = bench.pick_items("warm")
    cold = bench.pick_items("cold")
    allq = bench.pick_items("all")

    assert len(allq) == 26, f"goldset 条数变了：{len(allq)}"
    assert len(warm) == bench.N_WARM == 6
    assert len(cold) == bench.N_COLD == 20
    assert len(warm) + len(cold) == len(allq)

    warm_ids = {it["id"] for it in warm}
    cold_ids = {it["id"] for it in cold}
    assert not (warm_ids & cold_ids), f"热身与冷路径重叠：{warm_ids & cold_ids}"
    assert (warm_ids | cold_ids) == {it["id"] for it in allq}

    # 每条都得有 answer_substr，否则串场检测没有比对物
    assert all(it.get("answer_substr") for it in allq)


# ---------------------------------------------------------------- 代理绕过
def test_client_kwargs_disable_env_proxy():
    """**压测必须绕开环境代理**——这条是实测踩出来的，不是预防性洁癖。

    本机 shell 设了 `HTTP_PROXY=http://127.0.0.1:50297`（Clash），且没有 `NO_PROXY`。
    httpx 默认 `trust_env=True`，于是「压测本地服务」实际压的是代理：
    `/api/schema` 单发 22ms，10 并发 P50 直接变成 **13.5s**，
    而那个端点只是序列化一个 8KB 的内存字典。

    第一版读数差点让人得出「FastAPI 框架层很慢」的结论——归因错比没数据更糟。

    断言两个方向：kwargs 里必须写明 `trust_env=False`，
    且真的构造出来的 client 内部标志也是 False（防止有人只改注释不改参数）。
    """
    import httpx

    kw = bench.client_kwargs(conc=10, timeout_s=180.0)
    assert kw["trust_env"] is False, "缺 trust_env=False，请求会被环境代理接管"
    assert kw["follow_redirects"] is False

    cli = httpx.AsyncClient(**kw)
    try:
        assert cli._trust_env is False
    finally:
        import asyncio
        asyncio.run(cli.aclose())


def test_env_proxy_would_actually_hijack_requests(monkeypatch):
    """反向证明：开着 `trust_env` 时请求**确实**会被改道到代理。

    只断言「我们关掉了」是单边的——万一 httpx 哪天不再读 env，上一条就变成永真。
    两条合起来才算钉住：证明了改道真的会发生，也证明了我们躲开了它。

    实测（httpx 0.28.1）：代理不是通过 `_mounts` 的键名体现的，而是挂载 transport 的
    connection pool 上（`_pool._proxy_url`）。所以断言必须落在 pool 上，
    断言 `_mounts` 里有没有 "http" 键是错的——键是 URLPattern 对象。
    """
    import httpx

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:59999")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    target = httpx.URL("http://127.0.0.1:501/whatever")
    hijacked = httpx.Client(trust_env=True)
    direct = httpx.Client(**bench.client_kwargs(conc=2, timeout_s=5.0))
    try:
        pool_proxy = hijacked._transport_for_url(target)._pool._proxy_url
        assert pool_proxy is not None, "trust_env=True 时请求应被环境代理接管"
        assert "59999" in str(pool_proxy)
        # 我们用的 client 必须直连：pool 上没有代理
        # 直连时 transport 用的是普通 ConnectionPool，压根没有 _proxy_url 属性
        assert getattr(direct._transport_for_url(target)._pool, "_proxy_url", None) is None
    finally:
        hijacked.close()
        direct.close()


# ---------------------------------------------------------------- 控制台编码
def test_markdown_row_is_console_encodable():
    """markdown 行要能打到 GBK 控制台。

    仓库有静态闸门（`test_output_encoding.py`），但它只看源码里的字面量；
    这条走**真实 f-string 求值结果**——若哪天表头加了 `≈`/`⇒`，
    静态闸门看不见（它在 f-string 里），这条能抓。
    """
    s = {
        "qps": 12.3, "p50_ms": 10.0, "p95_ms": 20.0, "p99_ms": 30.0,
        "err_rate": 0.0,
    }
    row = bench.md_row("P1-A 缓存命中", 300, 10, s, "落库（store.py:_lock）")
    row.encode("gbk")            # 编不了会抛，即失败
    assert row.startswith("|") and row.endswith("|")


def test_cli_help_is_wellformed_and_needs_no_server():
    """`--help` 必须能跑：它证明 argparse 定义没问题，且不碰网络。

    压测脚本最难排查的失败是「参数写错但脚本照跑，只是测了别的东西」，
    所以参数表本身也要有一条用例。
    """
    with pytest.raises(SystemExit) as ei:
        bench.build_parser().parse_args(["--help"])
    assert ei.value.code == 0
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--path", "nonexistent"])

