"""
delta 拆分对照实验的自证（P1）。

## 为什么这个实验也要写测试

这个实验的作用是**给已有结论翻案或定案**。它一旦写错，错的方向恰好都是
「看起来支持我的归因」，所以比压测本身更需要先钉住：

1. **`noappend` 忘了 `return`** → 退化成 `base`，于是「拿掉落库 delta 也没变」，
   结论会变成「落库不是瓶颈」——与事实相反。
2. **`sleeplock` 的 `sleep` 写到了 `with` 外面** → 退化成 `sleepoutside`，
   「锁内 vs 锁外」差值恒为 0，结论会变成「锁归因被推翻」——也是反的。

这两种退化用「源码里出现过 `return None` / `with store._lock`」这类字符串断言
**抓不住**（字符串还在，位置错了）。所以 `patch_source()` 被单独抽出来，
本文件直接 `exec` 它并配一个假 `store`，断言的是**行为**：
原函数有没有被调用、被调用时锁有没有被持有。

判读侧同理：只断言「确认串行」是单边的——一个恒返回 True 的 `analyse`
也能过。必须同一批数据下让结论**翻转**一次。
"""
from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_exp():
    """从路径加载 evals/exp_delta_split.py（evals 不是包）。"""
    path = ROOT / "evals" / "exp_delta_split.py"
    spec = importlib.util.spec_from_file_location("exp_delta_split_p1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp = _load_exp()


# ============================================================================
# 注入体：行为级验证
# ============================================================================
class _FakeLock:
    """模拟 `RLock` 的重入语义，并暴露「此刻是否被持有」。"""

    def __init__(self) -> None:
        self.depth = 0

    @property
    def held(self) -> bool:
        return self.depth > 0

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, *exc):
        self.depth -= 1
        return False


def _exec_patch(mode: str, delay_ms: float = 30.0):
    """把 `patch_source()` 的产出 exec 进一个假 store 环境。

    返回 `(patched 函数, 调用记录, lock)`；`调用记录[i]["held_before"]`
    是真实实现**被调用的那一刻**锁是否被持有。
    """
    calls: list[dict] = []
    lock = _FakeLock()

    def real_append_turn(*a, **k):
        # 记录「进入真实写入之前」的锁状态：这正是 sleeplock 与 sleepoutside
        # 唯一不同的事实，也是整个实验的自变量。
        calls.append({"held_before": lock.held})
        with lock:                    # 真实 store.append_turn 内部就是 with _lock
            pass
        return "written"

    store = types.SimpleNamespace(append_turn=real_append_turn, _lock=None)
    ns = {
        "store": store,
        "time": time,
        # 让注入体里的 threading.RLock() 返回假锁，从而能观察到持有状态
        "threading": types.SimpleNamespace(RLock=lambda: lock),
    }
    exec(compile(exp.patch_source(mode, delay_ms), "<patch>", "exec"), ns)
    return store.append_turn, calls, lock                     # noqa: SLF001


def test_noappend_never_calls_the_real_write():
    """`noappend` 必须整段拿掉，而不是「变成 base」。"""
    patched, calls, _ = _exec_patch("noappend")
    t0 = time.perf_counter()
    assert patched("s", "q", "a", 0, False) is None
    assert time.perf_counter() - t0 < 0.02
    assert calls == []


def test_base_writes_and_does_not_sleep():
    """`base` 不注入：真实写入照跑，且没有额外延迟。"""
    patched, calls, _ = _exec_patch("base")
    t0 = time.perf_counter()
    assert patched("s", "q", "a", 0, False) == "written"
    assert time.perf_counter() - t0 < 0.02
    assert calls == [{"held_before": False}]


def test_sleeplock_holds_the_lock_when_the_real_write_runs():
    """决定性断言之一：锁内注入时，真实写入被调用那一刻锁必须已被持有。"""
    patched, calls, lock = _exec_patch("sleeplock", 30.0)
    t0 = time.perf_counter()
    assert patched("s", "q", "a", 0, False) == "written"
    elapsed = time.perf_counter() - t0
    assert calls == [{"held_before": True}]
    assert elapsed >= 0.025, f"锁内延迟没有被执行（{elapsed * 1000:.1f}ms）"
    assert lock.held is False, "注入结束后锁必须释放"


def test_sleepoutside_delays_the_same_but_outside_the_lock():
    """决定性断言之二：同样的延迟，真实写入时锁**未**被持有。

    与上一条合起来才是完整对照：只改「在不在临界区里」这一个变量。
    若两条断言都成立而 `analyse` 还是判「串行确认」，说明判读逻辑有问题；
    若只有这一条成立，说明注入写错了位置。
    """
    patched, calls, lock = _exec_patch("sleepoutside", 30.0)
    t0 = time.perf_counter()
    assert patched("s", "q", "a", 0, False) == "written"
    elapsed = time.perf_counter() - t0
    assert calls == [{"held_before": False}]
    assert elapsed >= 0.025, f"锁外延迟没有被执行（{elapsed * 1000:.1f}ms）"
    assert lock.held is False


def test_both_injecting_arms_spend_the_same_wall_time():
    """两臂的额外工作量必须等价，否则「斜率差」会混进工作量差。"""
    t_lock = t_out = None
    for mode, setter in (("sleeplock", "lock"), ("sleepoutside", "out")):
        patched, _, _ = _exec_patch(mode, 40.0)
        t0 = time.perf_counter()
        patched("s", "q", "a", 0, False)
        if setter == "lock":
            t_lock = time.perf_counter() - t0
        else:
            t_out = time.perf_counter() - t0
    assert abs(t_lock - t_out) < 0.02, f"两臂耗时差 {abs(t_lock - t_out) * 1000:.1f}ms 过大"


def test_patch_source_for_each_arm_is_syntactically_valid():
    """四臂的注入体都要能编译；拼错一个 `{` 会在起服务时才炸。"""
    for arm, spec in exp.ARMS.items():
        src = exp.patch_source(spec["mode"], 20.0)
        compile(src, f"<{arm}>", "exec")
        assert f"MODE = {spec['mode']!r}" in src
        assert "store.append_turn = patched" in src


def test_boot_source_embeds_patch_and_starts_uvicorn():
    src = exp.boot_source(ROOT, "sleeplock", 20.0, 8399)
    compile(src, "<boot>", "exec")
    assert "store.append_turn = patched" in src
    assert "port=8399" in src
    assert "src.api.app:app" in src
    # 注入必须发生在 uvicorn.run 之前：写到后面就没人 import 它了
    assert src.index("store.append_turn = patched") < src.index("uvicorn.run")
    # 生成的文件必须自带编码声明，否则中文注释在 cp936 环境下会炸
    assert src.lstrip().startswith("# -*- coding: utf-8 -*-")


# ============================================================================
# 判读：斜率、串行写模型、结论
# ============================================================================
def test_fit_slope_exact_on_perfect_line():
    slope, intercept, r2 = exp.fit_slope([1, 2, 4, 8], [12.0, 22.0, 42.0, 82.0])
    assert slope == pytest.approx(10.0)
    assert intercept == pytest.approx(2.0)
    assert r2 == pytest.approx(1.0)


def test_fit_slope_refuses_a_single_point():
    """一个点给不出斜率——必须返回 None，不能用 0 冒充「没有增长」。"""
    assert exp.fit_slope([3], [7.0]) == (None, None, None)
    assert exp.fit_slope([], []) == (None, None, None)


def test_arm_slope_handles_unsorted_and_sparse_levels():
    """档位不是等差的（1/2/5/10/20/40），斜率必须按最小二乘算，不能取端点差。"""
    pts = {40: 120.0, 1: 5.0, 10: 30.0, 2: 8.0, 20: 55.0, 5: 16.0}
    s = exp.arm_slope(pts)
    assert s == pytest.approx(exp.fit_slope([1, 2, 5, 10, 20, 40],
                                           [5.0, 8.0, 16.0, 30.0, 55.0, 120.0])[0])


# ---------------------------------------------------------------------------
# 实测数据：四臂跑出来的原始 delta P50（evals/exp_delta_split_main.json）
# 用真实数据做断言，比合成数据更能钉住判读逻辑。
# ---------------------------------------------------------------------------
REAL = {
    "base":         {1: 4.22, 2: 5.15, 5: 14.15, 10: 29.76, 20: 61.04, 40: 117.90},
    "noappend":     {1: 1.94, 2: 3.01, 5: 6.17, 10: 14.48, 20: 23.07, 40: 135.46},
    "sleeplock":    {1: 25.40, 2: 44.88, 4: 92.51, 8: 184.83},
    "sleepoutside": {1: 25.89, 2: 26.14, 4: 26.51, 8: 27.82},
}


def test_serial_write_model_fits_when_saturation_level_is_excluded():
    """拟合区间内：Δ(base−noappend) 随待排写入数线性增长，斜率≈单次写入耗时。"""
    m = exp.serial_write_model(REAL, fit_max=20)
    assert m["levels"] == [1, 2, 5, 10, 20]
    assert m["slope_ms_per_pending_write"] == pytest.approx(1.9, abs=0.2)
    assert m["r2"] > 0.95
    assert m["holds"] is True


def test_saturation_level_destroys_the_fit_and_flips_the_verdict():
    """**决定性断言**：同一批数据，只改拟合区间，结论必须翻转。

    c=40 时 noappend（服务端什么都不做）的 delta 比 base 还大，这不是服务端现象，
    而是读数出界。若判读不看区间、一路拟合到底，斜率会被这个杠杆点顶成负数，
    「落库导致增长」的结论就被自己推翻了——错在拟合方式，不在数据。
    """
    m_all = exp.serial_write_model(REAL, fit_max=None)
    assert m_all["slope_ms_per_pending_write"] < 0
    assert m_all["holds"] is False

    m_fit = exp.serial_write_model(REAL, fit_max=20)
    assert m_fit["holds"] is True


def test_find_anomalies_flags_the_level_where_noappend_exceeds_base():
    """越界的档位必须被显式点名，否则它会被当「瓶颈不是落库」的证据。"""
    anomalies = exp.find_anomalies(REAL)
    assert len(anomalies) == 1
    assert "c=40" in anomalies[0]
    assert "135.5ms > base 117.9ms" in anomalies[0]


def test_analyse_confirms_serialization_when_lock_arm_has_steeper_slope():
    r = exp.analyse(REAL, 20.0, fit_max=20, probe={"p50_ms": 1.81})
    assert r["serialization_confirmed"] is True
    assert r["lock_slope_gain"] == pytest.approx(22.7, abs=1.0)
    assert r["serial_write_model"]["holds"] is True
    assert r["serial_write_model"]["agrees_with_probe"] is True
    assert "临界区归因成立" in r["conclusion"]
    assert "互证" in r["conclusion"]


def test_analyse_rejects_serialization_when_both_arms_grow_alike():
    """**决定性断言**：只改 sleeplock 一列，临界区归因必须翻转。

    其余三臂与实测完全相同。若判读只看「delta 涨了」而不看斜率差，
    这里仍会判「串行确认」——那正是本次实验要避免的偷懒推理。
    """
    flat = {"base": REAL["base"], "noappend": REAL["noappend"],
            "sleeplock": {1: 45.0, 2: 48.0, 4: 53.0, 8: 58.0},
            "sleepoutside": REAL["sleepoutside"]}
    r = exp.analyse(flat, 20.0, fit_max=20, probe={"p50_ms": 1.81})
    assert r["serial_write_model"]["holds"] is True     # 落库仍是增长来源
    assert r["serialization_confirmed"] is False        # 但「临界区」归因被否
    assert "临界区归因**被推翻**" in r["conclusion"]


def test_analyse_flags_disagreement_with_independent_probe():
    """回归斜率与独立直测对不上时，必须明说模型不成立，不能只看 r² 好看。"""
    r = exp.analyse(REAL, 20.0, fit_max=20, probe={"p50_ms": 12.0})
    assert r["serial_write_model"]["agreement_ratio"] < 0.7
    assert r["serial_write_model"]["agrees_with_probe"] is False
    assert "对不上" in r["conclusion"]


def test_analyse_rejects_model_when_write_removal_changes_nothing():
    """拿掉落库后 delta 没有变化 = 串行写模型不成立。"""
    same = {"base": REAL["base"], "noappend": dict(REAL["base"]),
            "sleeplock": REAL["sleeplock"], "sleepoutside": REAL["sleepoutside"]}
    m = exp.serial_write_model(same, fit_max=20)
    assert m["slope_ms_per_pending_write"] == pytest.approx(0.0, abs=0.01)
    assert m["holds"] is False


def test_analyse_says_insufficient_evidence_instead_of_guessing():
    """缺臂时必须返回 None 并说「证据不足」，不能默认 False/True。"""
    r = exp.analyse({"base": REAL["base"], "noappend": REAL["noappend"]},
                    20.0, fit_max=20)
    assert r["serialization_confirmed"] is None
    assert "未验证" in r["conclusion"]

    r2 = exp.analyse({"sleeplock": REAL["sleeplock"],
                      "sleepoutside": REAL["sleepoutside"]}, 20.0)
    assert r2["serial_write_model"]["holds"] is None
    assert "无法检验" in r2["conclusion"]


def test_analyse_reports_excluded_levels_explicitly():
    """被 fit_max 剔掉的档位要落盘，保证「剔了哪些」可复查。"""
    r = exp.analyse(REAL, 20.0, fit_max=20)
    assert r["excluded_by_fit_max"]["base"] == [40]
    assert r["excluded_by_fit_max"]["sleeplock"] == []       # 最高只到 8


def test_write_share_is_computed_without_any_fitting():
    """逐档份额是不依赖拟合的读数：拟合再漂亮也可能错，份额不会。"""
    r = exp.analyse(REAL, 20.0, fit_max=20)
    share = r["write_share_by_concurrency"]
    assert set(share) == {"1", "2", "5", "10", "20"}
    assert 0.4 < share["10"] < 0.6
    assert "40" not in share                                 # 越界档不掺进来


def test_analyse_serializes_points_for_json():
    r = exp.analyse({"base": {10: 30.0, 1: 5.0}, "noappend": {1: 2.0, 10: 3.0}}, 20.0)
    assert r["points"]["base"] == {"1": 5.0, "10": 30.0}      # 键要是字符串且有序


# ============================================================================
# 表格与 CLI
# ============================================================================
def test_md_table_marks_levels_outside_the_fit_range():
    text = exp.md_table(REAL, fit_max=20)
    assert "| 20 |" in text
    assert "| 40* |" in text                                 # 越界要打星
    assert "117.9ms*" in text
    assert "不参与拟合" in text
    text.encode("gbk")                                        # Windows 控制台不能崩


def test_md_table_uses_dash_for_missing_levels():
    text = exp.md_table({"base": {1: 5.0, 2: 8.0}, "noappend": {1: 2.0}}, fit_max=None)
    assert "| 2 | 8.0ms | - |" in text


def test_cli_boot_only_prints_source_and_exits(capsys):
    assert exp.main(["--boot-only", "sleeplock", "--delay-ms", "12"]) == 0
    out = capsys.readouterr().out
    compile(out, "<stdout>", "exec")              # 打印出来的必须能直接当脚本跑
    assert "DELAY = 0.012" in out


def test_cli_defaults_cover_all_four_arms_and_truncate_fit():
    a = exp.build_parser().parse_args([])
    assert set(a.arm) == set(exp.ARMS)
    assert a.delay_ms == exp.DEFAULT_DELAY_MS
    assert a.fit_max == 20                       # c=40 默认不参与拟合
    assert a.n_full == 300                       # 与 T3 的 n 对齐，数字可比
    assert a.probe_n > 0                         # 独立探针默认开
    # 注入臂只爬到 8：更高并发会让 to_thread 的默认执行器变成第二个串行器
    assert max(exp.LEVELS_INJECT) <= 8
    assert max(exp.LEVELS_FULL) >= 40
