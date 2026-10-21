#!/usr/bin/env python
"""
对照实验：把 `delta = 客户端墙钟 - done.elapsed_ms` 拆开，看它到底来自哪。

## 要回答的问题

T3 把 P1-A（缓存命中）的天花板归因到「`done` 之后的落库被 `store.py:47` 的
进程级 `threading.Lock` 串行」。那是**推断**：`delta` 随并发近线性增长、
而服务端 `elapsed_ms` 恒定在 1-2ms，只能说明「`done` 之后发生了与并发成正比的等待」，
却区分不出这段等待是

  (a) 锁内串行（落库在排队），
  (b) 网络 / SSE 收尾，
  (c) 线程池本身排不上队。

本实验用「只改一件事」的四臂对照，把推断变成证据。

## 四臂：每臂一个独立服务进程，唯一变量是 `LT_EXP_MODE`

| arm | 注入 | 若 (a) 成立的预测 |
|---|---|---|
| `base` | 什么都不做 | delta 随 c 增长（复现 T3） |
| `noappend` | 整段跳过 `append_turn` | delta 变平 ⇒ 增长确实来自落库 |
| `sleeplock` | 锁内 `sleep(D)` | delta ~= base + D·(c-1)，斜率**变大** |
| `sleepoutside` | 锁外 `sleep(D)` | delta ~= base + D，斜率**不变** |

`sleeplock` vs `sleepoutside` 是**因果**证据：**同一份额外工作量**，
只有落在临界区里才会随并发被放大。若两者斜率差不出 D，「锁串行」的解释就是错的
——这时候该去查网络/收尾，而不是去优化数据库。

`base` vs `noappend` 是**定量**证据，而且它给出的不只是「变平/没变平」：
两者之差 `delta(c) = base(c) - noappend(c)` 应当等于「排在这条写前面的写有几个」x
「单次写入的锁占用时间」。若这个回归斜率能对上**独立测量**到的 `append_turn`
单次耗时，串行写模型就从「看起来像」升级为「量得上」。

## 判读必须分区间，否则会被杠杆点带跑

实测（见 `evals/exp_delta_split_main.json`）：

| 并发 | base | noappend |
|---|---|---|
| 20 | 61.0ms | 23.1ms |
| 40 | 117.9ms | **135.5ms** |

c=40 时「服务端什么都不做」的那一臂 delta **反而更大**。这不是服务端现象，
是读数已经出界（客户端单进程要同时消化 40 条 token 流）。
跨 c=1..40 做一条最小二乘，这个杠杆点会把 `noappend` 的斜率顶到比 `base` 还高，
于是结论翻转成「delta 增长与落库无关」——**错在拟合方式，不在数据**。
所以本脚本一律用 `--fit-max` 截断拟合区间，并把越界档位单独列出来。

## 为什么能不碰生产代码就注入

`app.py:104` 与 `app.py:209` 都写成 `store.append_turn(...)`（**模块属性访问**），
不是 `from ... import append_turn`。所以启动时替换 `src.api.store.append_turn`
即可命中全部调用点，仓库里的服务代码一行不改。

## 用法

    python evals/exp_delta_split.py                        # 跑全四臂 + 落库探针
    python evals/exp_delta_split.py --arm base             # 只跑一臂
    python evals/exp_delta_split.py --boot-only sleeplock  # 只打印注入源码
    python evals/exp_delta_split.py --probe-only           # 只量单次 append_turn
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_bench():
    """从路径加载 `evals/bench_concurrency.py`（evals 不是包）。"""
    path = ROOT / "evals" / "bench_concurrency.py"
    spec = importlib.util.spec_from_file_location("bench_concurrency_p1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()


# ============================================================================
# 设计常量
# ============================================================================
ARMS: dict[str, dict] = {
    "base":         {"mode": "base",         "injects_delay": False,
                     "purpose": "复现 T3 的 delta 曲线（对照基线）"},
    "noappend":     {"mode": "noappend",     "injects_delay": False,
                     "purpose": "把落库整段拿掉：base-noappend 就是落库的贡献"},
    "sleeplock":    {"mode": "sleeplock",    "injects_delay": True,
                     "purpose": "锁内加固定延迟：斜率应增加约 D ms/并发"},
    "sleepoutside": {"mode": "sleepoutside", "injects_delay": True,
                     "purpose": "锁外加同样延迟：斜率应基本不变"},
}

# base / noappend 爬到 40 是为了复现 T3 的尾部；
# 注入臂只爬到 8：`asyncio.to_thread` 的默认执行器只有
# min(32, cpu+4) 个线程，注入 sleep 会占住线程，高并发下执行器本身
# 会变成第二个串行器，把「锁」的信号搅浑。这是刻意避开第 (c) 种混淆。
LEVELS_FULL = [1, 2, 5, 10, 20, 40]
LEVELS_INJECT = [1, 2, 4, 8]

DEFAULT_DELAY_MS = 20.0
# c=40 已在读数有效区间之外（见模块文档的实测对比），默认不参与拟合
DEFAULT_FIT_MAX = 20


# ============================================================================
# 纯函数（可单测，不碰网络）
# ============================================================================
def fit_slope(xs: list[float], ys: list[float]) -> tuple[float | None, float | None, float | None]:
    """最小二乘拟合 `y = intercept + slope·x`，返回 `(slope, intercept, r2)`。

    点少于 2 个返回 `(None, None, None)`——**不给斜率就明说拿不到**，
    不要用 0 冒充「没有增长」。
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return None, None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return slope, intercept, r2


def arm_slope(points: dict[int, float]) -> float | None:
    """一条 arm 的 delta(c) 曲线斜率（ms / 每单位并发）。"""
    xs = sorted(points)
    if len(xs) < 2:
        return None
    return fit_slope(xs, [points[x] for x in xs])[0]


def serial_write_model(deltas: dict[str, dict[int, float]],
                       fit_max: int | None = None) -> dict:
    """拟合 `delta(c) = base(c) - noappend(c)`，检验「串行写」这个模型。

    模型：`done` 之后写一条历史，写之前要抢一把进程级锁。并发 c 时，
    一条写前面平均排着 (c-1) 条（闭循环下队列始终接近满），所以

        delta(c) ~= t_write + (c-1)·t_lock

    `t_lock` 就是单次写入的锁占用时间，`t_write` 是 c=1、无需排队时的单次耗时。
    把 delta 对 (c-1) 做回归，斜率应当**等于独立测到的 `append_turn` 单次耗时**。
    对上，串行写模型成立；对不上，说明「锁」这个词用错了。

    只保留两臂都测过的档位，且受 `fit_max` 截断——见模块文档里 c=40 的反例。
    """
    base = deltas.get("base", {})
    noap = deltas.get("noappend", {})
    common = sorted(set(base) & set(noap))
    if fit_max is not None:
        common = [c for c in common if c <= fit_max]
    if len(common) < 2:
        return {"levels": common, "delta_ms": {},
                "slope_ms_per_pending_write": None, "intercept_ms": None,
                "r2": None, "holds": None}

    xs = [c - 1 for c in common]
    ys = [base[c] - noap[c] for c in common]
    slope, intercept, r2 = fit_slope(xs, ys)
    holds = bool(slope is not None and slope > 0.2 and (r2 or 0) >= 0.8)
    return {
        "levels": common,
        "delta_ms": {str(c): round(base[c] - noap[c], 3) for c in common},
        "slope_ms_per_pending_write": None if slope is None else round(slope, 3),
        "intercept_ms": None if intercept is None else round(intercept, 3),
        "r2": None if r2 is None else round(r2, 4),
        "holds": holds,
    }


def find_anomalies(deltas: dict[str, dict[int, float]]) -> list[str]:
    """找出「服务端工作量更少、delta 反而更大」的档位。

    这不是服务端现象，是**读数出界**：量不到的档位必须显式列出来，
    否则它们会被当成「瓶颈不是落库」的证据。
    """
    base = deltas.get("base", {})
    noap = deltas.get("noappend", {})
    out = []
    for c in sorted(set(base) & set(noap)):
        if noap[c] > base[c] * 1.05:
            out.append(
                f"c={c}: noappend {noap[c]:.1f}ms > base {base[c]:.1f}ms "
                f"（去掉落库后 delta 反而变大 -> 该档读数已不在测服务端，剔除拟合）")
    return out


def analyse(deltas: dict[str, dict[int, float]], delay_ms: float,
            fit_max: int | None = DEFAULT_FIT_MAX,
            probe: dict | None = None) -> dict:
    """把四臂的 delta(c) 曲线读成三条可证伪的结论。

    输入形如 `{"base": {1: 4.2, 2: 5.2, ...}, "noappend": {...}, ...}`，
    值是各档的 delta P50（ms）。

    三条判据分别证不同的事，任何一条为 `None` 就明说「证据不足」，不猜：

    1. `serial_write_model`：`delta = base - noappend` 对 (c-1) 的回归斜率，
       是否等于独立测量的单次写入耗时。证的是「落库按串行排队解释得通」。
    2. `serialization_confirmed`：`sleeplock` 与 `sleepoutside` 的斜率差
       是否达到注入延迟量级的一半。证的是**因果**——放大来自「落在临界区里」，
       而不是「多做了 D 毫秒的活」。
    3. `write_share_by_concurrency`：逐档看落库占 delta 的比例。
       这是唯一不依赖拟合的读数，用来防止「拟合漂亮但其实是错的」。
    """
    def truncate(points: dict[int, float]) -> dict[int, float]:
        if fit_max is None:
            return dict(points)
        return {c: v for c, v in points.items() if c <= fit_max}

    fit = {arm: truncate(pts) for arm, pts in deltas.items()}

    s_base = arm_slope(fit.get("base", {}))
    s_noap = arm_slope(fit.get("noappend", {}))
    s_lock = arm_slope(fit.get("sleeplock", {}))
    s_out = arm_slope(fit.get("sleepoutside", {}))

    out: dict = {
        "delay_ms": delay_ms,
        "fit_max": fit_max,
        "slopes": {"base": s_base, "noappend": s_noap,
                   "sleeplock": s_lock, "sleepoutside": s_out},
        "points": {k: {str(c): v for c, v in sorted(v.items())} for k, v in deltas.items()},
        "excluded_by_fit_max": {
            k: sorted(c for c in v if fit_max is not None and c > fit_max)
            for k, v in deltas.items()
        },
        "anomalies": find_anomalies(deltas),
    }

    # ---- 判据 1：串行写模型的定量检验 ----
    model = serial_write_model(deltas, fit_max)
    if probe and model.get("slope_ms_per_pending_write") and probe.get("p50_ms"):
        model["probe_p50_ms"] = round(probe["p50_ms"], 3)
        model["agreement_ratio"] = round(
            model["slope_ms_per_pending_write"] / probe["p50_ms"], 3)
        # 两个独立方法（并发排队回归 vs 单线程直测）落在 ±30% 内即视为互证
        model["agrees_with_probe"] = bool(0.7 <= model["agreement_ratio"] <= 1.3)
    else:
        model["probe_p50_ms"] = None
        model["agreement_ratio"] = None
        model["agrees_with_probe"] = None
    out["serial_write_model"] = model

    # ---- 判据 2：临界区的因果检验 ----
    if s_lock is not None and s_out is not None:
        gain = s_lock - s_out
        thr = 0.5 * delay_ms
        out["lock_slope_gain"] = round(gain, 3)
        out["serialization_threshold"] = round(thr, 3)
        out["serialization_confirmed"] = bool(gain >= thr)
    else:
        out["lock_slope_gain"] = None
        out["serialization_threshold"] = round(0.5 * delay_ms, 3)
        out["serialization_confirmed"] = None

    # ---- 判据 3：不依赖拟合的逐档份额 ----
    share = {}
    for c in sorted(set(fit.get("base", {})) & set(fit.get("noappend", {}))):
        b = fit["base"][c]
        if b:
            share[str(c)] = round(1 - fit["noappend"][c] / b, 3)
    out["write_share_by_concurrency"] = share

    out["conclusion"] = _conclusion(out)
    return out


def _conclusion(r: dict) -> str:
    """把三条判据读成人话。任一为 None 时明确说「证据不足」，不猜。"""
    m = r["serial_write_model"]
    parts = []

    if m.get("slope_ms_per_pending_write") is None:
        parts.append("缺少 base/noappend 对比，串行写模型无法检验")
    elif m["holds"]:
        txt = (f"落库按串行排队解释得通：delta(base-noappend) 随待排写入数增长 "
               f"{m['slope_ms_per_pending_write']} ms/条（R2={m['r2']}，"
               f"档位 {m['levels']}）")
        if m.get("agrees_with_probe"):
            txt += (f"，与独立直测 append_turn {m['probe_p50_ms']}ms 相差 "
                    f"{abs(m['agreement_ratio'] - 1) * 100:.0f}% -> 互证")
        elif m.get("agrees_with_probe") is False:
            txt += (f"，但独立直测 append_turn 是 {m['probe_p50_ms']}ms，"
                    f"比值 {m['agreement_ratio']} -> **对不上，模型不成立**")
        parts.append(txt)
    else:
        parts.append(f"串行写模型**不成立**（斜率 {m.get('slope_ms_per_pending_write')}，"
                     f"R2={m.get('r2')}）")

    if r["serialization_confirmed"] is None:
        parts.append("缺少 sleeplock/sleepoutside 对比，临界区归因未验证")
    elif r["serialization_confirmed"]:
        parts.append(
            f"临界区归因成立（锁内加 {r['delay_ms']}ms 使斜率多涨 "
            f"{r['lock_slope_gain']} ms/并发，阈值 {r['serialization_threshold']}）")
    else:
        parts.append(
            f"临界区归因**被推翻**（锁内与锁外斜率差仅 {r['lock_slope_gain']}，"
            f"低于阈值 {r['serialization_threshold']}）")

    shares = list(r["write_share_by_concurrency"].values())
    if shares:
        parts.append(f"落库占 delta 的 {min(shares) * 100:.0f}%–{max(shares) * 100:.0f}%"
                     f"（拟合区间内，逐档见 write_share_by_concurrency）")
    if r["anomalies"]:
        parts.append("注意" + "；".join(r["anomalies"]))
    return "；".join(parts) + "。"


def patch_source(mode: str, delay_ms: float) -> str:
    """只生成 `append_turn` 的注入体（不含启动服务那段）。

    ## 为什么单独抽出来

    注入逻辑本身是可以出错的，而且错了会**静默给出好看的数**：
    比如 `noappend` 分支忘了 `return`，它就会退化成 `base`；
    比如 `sleeplock` 的 `sleep` 写到了 `with` 外面，它就会退化成 `sleepoutside`，
    于是「锁内 vs 锁外」这组对照变成「锁外 vs 锁外」，差值恒为 0 ——
    看起来是「锁归因被推翻」，其实是代码写错了。

    抽出来之后测试可以直接 `exec` 这段源码、配一个假 `store` 跑一遍，
    断言的是**行为**（原函数有没有被调用、调用时锁有没有被持有），
    而不是「源码里出现过某个字符串」。只断言字符串挡不住上面两种退化。
    """
    delay_s = delay_ms / 1000.0
    return f'''MODE = {mode!r}
DELAY = {delay_s!r}
orig = store.append_turn

# RLock：`store.append_turn` 内部自己 `with _lock`，注入的 sleep 若也拿同一把
# 非重入锁，同线程二次 acquire 会死锁。RLock 让「sleep + 真实写入」共处一个临界区。
store._lock = threading.RLock()


def patched(*args, **kwargs):
    if MODE == "noappend":
        return None                      # 整段拿掉，但保持调用点不变
    if MODE == "sleeplock":
        with store._lock:                # 与真实写入同一临界区
            time.sleep(DELAY)
            return orig(*args, **kwargs)
    if MODE == "sleepoutside":
        time.sleep(DELAY)                # 同样的额外工作量，但在临界区外
        return orig(*args, **kwargs)
    return orig(*args, **kwargs)         # base：结构与其他臂一致，只是不注入


store.append_turn = patched
'''


def boot_source(root: Path, mode: str, delay_ms: float, port: int) -> str:
    """生成「带故障注入的服务启动器」完整源码。

    生成而不是静态文件，是为了让「注入什么」和「跑的实验」写在同一处：
    静态文件容易改了参数忘了改注入体，跑出来还是上一组数。
    """
    return f'''# -*- coding: utf-8 -*-
"""自动生成，勿手改：本文件由 evals/exp_delta_split.py 写进 .tmp/ 后执行。

arm={mode}  delay={delay_ms}ms  port={port}
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path({str(root)!r})
sys.path.insert(0, str(ROOT))

from src.api import store

{patch_source(mode, delay_ms)}

import uvicorn

uvicorn.run(
    "src.api.app:app",
    host="127.0.0.1",
    port={port},
    log_level="warning",
    timeout_keep_alive=120,
)
'''


def md_table(deltas: dict[str, dict[int, float]],
             fit_max: int | None = DEFAULT_FIT_MAX) -> str:
    """四臂并排的 markdown 表；拟合区间之外的档位后面标注 `*`。"""
    levels = sorted({c for d in deltas.values() for c in d})
    arms = [a for a in ARMS if a in deltas]
    lines = ["| 并发 | " + " | ".join(arms) + " |",
             "|---|" + "---|" * len(arms)]
    for c in levels:
        mark = "*" if (fit_max is not None and c > fit_max) else ""
        cells = []
        for a in arms:
            v = deltas[a].get(c)
            cells.append("-" if v is None else f"{v:.1f}ms{mark}")
        lines.append(f"| {c}{mark} | " + " | ".join(cells) + " |")
    if any(fit_max is not None and c > fit_max for c in levels):
        lines.append("")
        lines.append(f"`*` = 超出有效读数区间（fit_max={fit_max}），不参与拟合。")
    return "\n".join(lines)


# ============================================================================
# 独立探针：直接量一次 append_turn 有多贵
# ============================================================================
def probe_append(n: int = 50, session_prefix: str = "lt-exp-probe") -> dict:
    """单线程连续写 n 次 `store.append_turn`，报 P50 / P95。

    这是**独立于压测**的第二个测量方法：不经过 HTTP、不经过 SSE、不经过事件循环。
    它量到的就是「一次写入占用锁多久」，正好是串行写模型里那个斜率。

    ⚠️ 会往 `chat_turns` 真实写入 n 行（session 前缀 `lt-exp-probe` 唯一化，
    不影响任何既有会话）。这比「假设一个耗时」诚实得多。
    """
    from src.api import store

    lat: list[float] = []
    for i in range(n):
        sid = f"{session_prefix}-{time.time_ns()}-{i}"
        t0 = time.perf_counter()
        store.append_turn(sid, "probe-q", "probe-a", 0, False)
        lat.append((time.perf_counter() - t0) * 1000)
    return {
        "n": n,
        "p50_ms": round(bench.percentile(lat, 50), 3),
        "p95_ms": round(bench.percentile(lat, 95), 3),
        "mean_ms": round(sum(lat) / len(lat), 3),
    }


# ============================================================================
# 运行时
# ============================================================================
def _health_ok(base_url: str, need_engine: bool = True) -> bool:
    """绕开环境代理的探活（本机 Clash 会把 localhost 请求劫走）。

    光看 200 不够：`/health` 在引擎冷启动期间就返回 200，
    但 `/api/chat` 此时抛 503（`app.py:228-229`）。必须等 `engine_ready`。
    不等就开打，热身那几条会拿到 503，缓存永远进不去，
    后面所有 delta 都在测冷路径 —— 而数字看起来依然「有值」。
    """
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with op.open(base_url.rstrip("/") + "/health", timeout=5) as r:
            if r.status != 200:
                return False
            body = json.loads(r.read().decode("utf-8"))
    except Exception:                             # noqa: BLE001
        return False
    return bool(body.get("engine_ready")) if need_engine else True


def _wait_health(base_url: str, timeout_s: float, need_engine: bool = True) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _health_ok(base_url, need_engine):
            return True
        time.sleep(1.0)
    return False


def _start_arm(mode: str, delay_ms: float, port: int, py: str) -> subprocess.Popen:
    """把注入源码写到 .tmp/ 再执行。

    不直接 `python -c "<源码>"`：源码含中文注释，Windows 命令行以 cp936 传参
    会与 UTF-8 源码不匹配；写成文件由解释器按 UTF-8 读取，编码问题一次性消掉。
    """
    boot = ROOT / ".tmp" / f"exp_boot_{mode}.py"
    boot.parent.mkdir(parents=True, exist_ok=True)
    boot.write_text(boot_source(ROOT, mode, delay_ms, port), encoding="utf-8")
    env = os.environ.copy()
    env["LT_EXP_MODE"] = mode
    env["LT_EXP_DELAY_MS"] = str(delay_ms)
    env["PYTHONIOENCODING"] = "utf-8"
    env["RERANK_ENABLED"] = "false"        # 四臂统一关掉，压热身的耗时，不影响 delta
    return subprocess.Popen([py, str(boot)], cwd=str(ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")


def _stop(proc: subprocess.Popen, port: int) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill()
    # 等端口真的释放，否则下一臂起不来（表现为 connection reset 混进错误率）
    t0 = time.time()
    while time.time() - t0 < 30 and _health_ok(f"http://127.0.0.1:{port}", need_engine=False):
        time.sleep(0.5)


def _warm(url: str, n_warm: int, timeout_s: float) -> tuple[bool, str]:
    """热身 + 缓存闸门：必须用新 session 再问一次并看到 `cached=true`。

    `cache_set` 要求 `fresh_session ∧ answer 非空 ∧ citations 非空 ∧
    hallucination_passed` 四条同时成立（`app.py:211-212`），最后一条在响应里看不见。
    只断言 HTTP 200 会把「没进缓存」误当热身成功，随后所有 delta 都在测冷路径。
    """
    items = bench.pick_items("warm")[:n_warm]
    for it in items:
        rec = asyncio.run(bench._single(url, it, f"lt-exp-warm-{it['id']}-{time.time_ns()}",
                                        timeout_s))
        print(f"  热身 {it['id']}: {rec['outcome']}  {rec['wall_ms']:.0f}ms"
              f"  err={rec['error']}")
    miss = []
    for it in items:
        rec = asyncio.run(bench._single(url, it, f"lt-exp-ver-{it['id']}-{time.time_ns()}",
                                        timeout_s))
        if rec["cached"] is not True:
            miss.append(it["id"])
    if miss:
        return False, f"缓存闸门未过：{miss} 未命中（fakeredis 随进程消失，换臂必须重热）"
    return True, "ok"


def run_arm(arm: str, args) -> dict:
    """跑一臂：起服务 -> 热身 -> 各档压测 -> 关服务。"""
    spec = ARMS[arm]
    base_url = f"http://127.0.0.1:{args.port}"
    url = base_url + "/api/chat"          # bench 的 run_http / _single 要的已是完整端点
    levels = LEVELS_INJECT if spec["injects_delay"] else LEVELS_FULL
    n = args.n_inject if spec["injects_delay"] else args.n_full

    print(f"\n{'=' * 70}")
    print(f"[arm {arm}] {spec['purpose']}")
    print(f"          注入={spec['mode']}  延迟={args.delay_ms}ms  "
          f"档位={levels}  每档 {n} 条")
    print(f"{'=' * 70}")

    proc = _start_arm(spec["mode"], args.delay_ms, args.port, sys.executable)
    try:
        if not _wait_health(base_url, args.start_timeout):
            tail = ""
            try:
                proc.terminate()
                tail = (proc.stdout.read() or "")[-2000:] if proc.stdout else ""
            except Exception:                     # noqa: BLE001
                pass
            raise RuntimeError(f"[arm {arm}] 服务未就绪（{args.start_timeout}s）\n{tail}")

        ok, why = _warm(url, args.warm, args.timeout)
        if not ok:
            raise RuntimeError(f"[arm {arm}] {why}")

        items = bench.pick_items("warm")[:1]
        deltas: dict[int, float] = {}
        detail: dict[int, dict] = {}
        for conc in levels:
            records, wall_s = asyncio.run(
                bench.run_http(url, items, n, conc, args.timeout, progress=False)
            )
            s = bench.summarize(records, wall_s)
            hit = sum(1 for r in records if r.get("cached") is True) / max(len(records), 1)
            d = s.get("delta_p50_ms")
            el = [r["elapsed_ms"] for r in records
                  if r.get("elapsed_ms") is not None and r.get("ok")]
            el_p50 = bench.percentile(el, 50) if el else None
            detail[conc] = {
                "n": len(records), "qps": s["qps"], "p50_ms": s.get("p50_ms"),
                "delta_p50_ms": d, "elapsed_p50_ms": el_p50,
                "cached_rate": round(hit, 4), "err_rate": s["err_rate"],
            }
            print(f"  c={conc:<3} n={len(records):<4} QPS {s['qps']:>7.2f}  "
                  f"P50 {None if s['p50_ms'] is None else round(s['p50_ms'], 1)}ms  "
                  f"deltaP50 {None if d is None else round(d, 1)}ms  "
                  f"elapsedP50 {None if el_p50 is None else round(el_p50, 1)}ms  "
                  f"hit {hit * 100:.0f}%  err {s['err_rate'] * 100:.1f}%")
            if hit < 1.0:
                raise RuntimeError(
                    f"[arm {arm}] c={conc} 缓存命中率 {hit * 100:.0f}% < 100%，该档 delta 作废")
            if d is not None:
                deltas[conc] = d
        return {"arm": arm, "deltas": deltas, "detail": detail,
                "n": n, "levels": levels}
    finally:
        _stop(proc, args.port)


# ============================================================================
# CLI
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="delta 拆分对照实验（P1 缓存命中路径）")
    p.add_argument("--arm", nargs="+", default=list(ARMS), choices=list(ARMS),
                   help="跑哪几臂（默认全四臂）")
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--delay-ms", type=float, default=DEFAULT_DELAY_MS,
                   help="sleeplock / sleepoutside 注入的延迟")
    p.add_argument("--fit-max", type=int, default=DEFAULT_FIT_MAX,
                   help="只用 <= 该并发的档位做拟合（0=不截断）；c=40 时读数会出界")
    p.add_argument("--n-full", type=int, default=300, help="base/noappend 每档请求数")
    p.add_argument("--n-inject", type=int, default=40, help="注入臂每档请求数")
    p.add_argument("--warm", type=int, default=2, help="每臂热几条问题（走 LLM，有成本）")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--start-timeout", type=float, default=240.0)
    p.add_argument("--probe-n", type=int, default=50,
                   help="落库探针连续写几次（会往 chat_turns 写真实行）")
    p.add_argument("--no-probe", action="store_true", help="跳过落库探针")
    p.add_argument("--probe-only", action="store_true", help="只跑探针，不启服务")
    p.add_argument("--tag", default="")
    p.add_argument("--boot-only", choices=list(ARMS), nargs="?",
                   help="只打印该臂的注入源码后退出（不启服务）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fit_max = None if args.fit_max <= 0 else args.fit_max

    if args.boot_only:
        print(boot_source(ROOT, ARMS[args.boot_only]["mode"], args.delay_ms, args.port))
        return 0

    probe = None
    if not args.no_probe:
        print(f"[探针] 单线程直写 append_turn {args.probe_n} 次（独立于压测的第二把尺）")
        probe = probe_append(args.probe_n)
        print(f"       P50 {probe['p50_ms']}ms  P95 {probe['p95_ms']}ms  "
              f"mean {probe['mean_ms']}ms")
        if args.probe_only:
            return 0

    results = {}
    for arm in args.arm:
        results[arm] = run_arm(arm, args)

    deltas = {a: r["deltas"] for a, r in results.items()}
    print("\n" + "=" * 70)
    print(f"## 四臂 delta P50（ms）  拟合区间 c <= {fit_max}")
    print(md_table(deltas, fit_max))
    verdict = analyse(deltas, args.delay_ms, fit_max=fit_max, probe=probe)

    print("\n## 判读")
    for k in ("base", "noappend", "sleeplock", "sleepoutside"):
        v = verdict["slopes"].get(k)
        print(f"  斜率 {k:<13} = {'-' if v is None else f'{v:.3f}'} ms / 每单位并发")
    m = verdict["serial_write_model"]
    print(f"  串行写模型          : {m.get('slope_ms_per_pending_write')} ms/每多一条待排写入"
          f"  (R2={m.get('r2')}, 截距 {m.get('intercept_ms')}ms, 档位 {m.get('levels')})"
          f"  -> holds={m.get('holds')}")
    print(f"  独立直测 append_turn: {m.get('probe_p50_ms')} ms"
          f"  比值 {m.get('agreement_ratio')} -> agrees={m.get('agrees_with_probe')}")
    print(f"  锁内减锁外          : {verdict['lock_slope_gain']} ms/并发 "
          f"(阈值 {verdict['serialization_threshold']}) "
          f"-> serialization_confirmed={verdict['serialization_confirmed']}")
    print(f"  落库占 delta 份额   : {verdict['write_share_by_concurrency']}")
    for a in verdict["anomalies"]:
        print(f"  [ANOMALY] {a}")
    print(f"\n结论：{verdict['conclusion']}")

    out = ROOT / "evals" / f"exp_delta_split{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps({
        "meta": {"delay_ms": args.delay_ms, "fit_max": fit_max,
                 "n_full": args.n_full, "n_inject": args.n_inject,
                 "warm": args.warm, "arms": args.arm, "port": args.port,
                 "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "tag": args.tag},
        "probe_append": probe,
        "deltas": {a: {str(c): v for c, v in d.items()} for a, d in deltas.items()},
        "detail": {a: {str(c): v for c, v in sorted(r["detail"].items())}
                   for a, r in results.items()},
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
