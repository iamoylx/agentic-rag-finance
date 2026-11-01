"""测试：容器的监听地址由 API_HOST 决定，生产镜像默认绑 0.0.0.0。

## 为什么单独为这件事写一个测试文件

真发生过一次事故：`docker ps` 显示 `agentic-rag-finance-api-1 (healthy)`，
宿主机 `curl 127.0.0.1:8300/health` 一个字节都收不到。

根因不在端口映射，而在**进程绑的地址**：
  - `scripts/run_api.py` 把 host 写死成 `127.0.0.1`，进程只监听**容器内的** loopback；
  - compose 里写的 `8300:8300` 是把宿主端口转发到容器网卡地址，转发进来的包
    没人接，于是天然失败；
  - 而 healthcheck 是在**容器内**打 `127.0.0.1:8300`，永远是通的 —— 两件事
    正交，所以监控全绿、服务全死。这条比 bug 本身更值得记住。

## 为什么不用「抽个纯函数再断言它」的测法

把地址解析抽成 `bind_addr()` 然后断言它返回 `"0.0.0.0"`，只能证明
**函数正确**，证明不了**启动路径真的用了它**。M6 那次就是这么翻车的：
删掉主流程里那行调用之后，9 条叶子用例全绿，只有 2 条接线用例红。

所以这里用 `runpy` 以 `__main__` 身份执行 `scripts/run_api.py` **文件本身**，
再拦下 `uvicorn.run` 看它实际收到的 kwargs。谁把 `os.getenv` 换成硬编码、
谁把 0.0.0.0 又改回 127.0.0.1、谁把 ENV 从 Dockerfile 里删掉，这里都会红。

## 为什么不干脆在 CI 里起全栈 api 再 curl 宿主机

P2 是这么做的（`.github/workflows/docker.yml` 的 compose job 里有
`curl http://127.0.0.1:8501/api/health`），因为 P2 的镜像自带合成 fixture 库。
P1 做不到：api 要 `./models` 里 4.3 GB 权重 + `./data` 里 10014 个分块。
实测过空卷启动，容器直接崩在 BM25 建库：

    File "src/retrieval/bm25.py", line 95, in build
      self._bm25 = BM25Okapi(self._tokenized, k1=self.k1, b=self.b)
    ZeroDivisionError: division by zero        # corpus_size == 0

所以「全栈起来再 curl」在 P1 的 CI 里是死路，dev 机上也复现不了（有卷）。
这个文件的定位就是替代方案：**在启动路径这一层把契约钉死**。
"""
from __future__ import annotations

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_API = PROJECT_ROOT / "scripts" / "run_api.py"

# 生产容器里必须绑的通配地址。绑 127.0.0.1 时 docker 的端口映射收不到任何东西。
WILDCARD = "0.0.0.0"
# 本机开发的默认值。刻意**不**是通配地址：本机跑 `python scripts/run_api.py`
# 不该无意识地把 8300 暴露到局域网。
LOOPBACK = "127.0.0.1"


def _read_repo_file(name: str) -> str:
    """读仓库根目录下的文件，缺失时报出**可操作**的错误。

    为什么需要这个壳：下面几条契约用例要读 Dockerfile / docker-compose.yml，
    而 CI 是在 `p1-rag:test` 镜像**内部**跑 pytest 的。镜像的 test 阶段必须
    `COPY Dockerfile docker-compose.yml ./`（见 Dockerfile 对应位置），否则
    容器里会红在一个与被测行为完全无关的 FileNotFoundError 上。
    刻意**不**写成「文件不存在就 skip」——那样闸门会在容器里静默失效，
    本机绿、CI 里其实什么都没验。
    """
    path = PROJECT_ROOT / name
    assert path.exists(), (
        f"找不到 {name}：镜像的 test 阶段必须 COPY 它（见 Dockerfile 的 test stage），"
        "这个文件是容器监听地址契约的一部分"
    )
    return path.read_text(encoding="utf-8")


def _capture_uvicorn_run(monkeypatch) -> dict:
    """以 __main__ 身份执行 scripts/run_api.py，返回 uvicorn.run 实际收到的参数。

    语法：monkeypatch 掉 `uvicorn.run` 再 `runpy.run_path(..., run_name="__main__")`。
    原理：run_api.py 末尾是 `if __name__ == "__main__": uvicorn.run(...)`，
          用 run_name="__main__" 才能让那个守卫成立，从而跑到真实的调用行。
          脚本内部 `import uvicorn` 拿到的是 sys.modules 里同一个模块对象，
          所以 patch 模块属性即可拦住调用（不需要改脚本）。
    """
    import uvicorn

    seen: dict = {}

    def fake_run(*args, **kwargs):
        seen["args"] = args
        seen.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    runpy.run_path(str(RUN_API), run_name="__main__")
    assert seen, "uvicorn.run 没有被调用 —— 启动守卫或调用行被改坏了"
    return seen


# =====================================================================
# 1. 生产容器口径：Dockerfile 会注入 API_HOST=0.0.0.0
# =====================================================================


def test_container_env_binds_wildcard(monkeypatch) -> None:
    """API_HOST=0.0.0.0 时，uvicorn 必须真的按通配地址起。"""
    monkeypatch.setenv("API_HOST", WILDCARD)
    seen = _capture_uvicorn_run(monkeypatch)
    assert seen["host"] == WILDCARD, (
        f"容器内必须绑 {WILDCARD}，实际 {seen['host']!r} —— "
        "绑 loopback 会让 docker 的端口映射收不到任何东西"
    )


# =====================================================================
# 2. 本机开发口径：不设环境变量时仍是 loopback
# =====================================================================


def test_local_env_keeps_loopback(monkeypatch) -> None:
    """不设 API_HOST 时必须回落 127.0.0.1。

    这条同时防住一种「修 bug 修过头」的改法：为了保险干脆把默认值写死成
    0.0.0.0。那样容器是通了，但本机开发会无意识地对局域网开放 8300。
    """
    monkeypatch.delenv("API_HOST", raising=False)
    seen = _capture_uvicorn_run(monkeypatch)
    assert seen["host"] == LOOPBACK, (
        f"未设 API_HOST 时应回落 {LOOPBACK}，实际 {seen['host']!r}"
    )


# =====================================================================
# 3. 端口同样走环境变量
# =====================================================================


def test_port_env_is_respected_and_typed(monkeypatch) -> None:
    """API_PORT 必须被读入，且必须是 int（uvicorn 不接受字符串端口）。"""
    monkeypatch.setenv("API_PORT", "18300")
    seen = _capture_uvicorn_run(monkeypatch)
    assert seen["port"] == 18300
    assert isinstance(seen["port"], int), "端口必须是 int，字符串会让 uvicorn 起不来"


def test_default_port_matches_compose_mapping(monkeypatch) -> None:
    """未设 API_PORT 时默认 8300 —— 与 compose / Dockerfile 的映射保持一致。"""
    monkeypatch.delenv("API_PORT", raising=False)
    seen = _capture_uvicorn_run(monkeypatch)
    assert seen["port"] == 8300


# =====================================================================
# 4. 镜像契约：Dockerfile 的 ENV 与 CMD 是这套参数的唯一来源
# =====================================================================
# 容器里 compose 的 api 服务**没有**在 environment: 里写 API_HOST
# （写了反而会让「镜像自带正确默认值」这件事失去保障）。也就是说
# 一旦 Dockerfile 里那行 ENV 被删，运行期就会静默回落到 loopback，
# 又变回「healthy 但连不上」。所以这一层契约必须单独钉住。


def test_dockerfile_bakes_wildcard_env() -> None:
    """Dockerfile 必须自带 API_HOST=0.0.0.0 与 API_PORT=8300。"""
    text = _read_repo_file("Dockerfile")
    assert f"API_HOST={WILDCARD}" in text, "Dockerfile 丢了 ENV API_HOST=0.0.0.0"
    assert "API_PORT=8300" in text, "Dockerfile 丢了 ENV API_PORT=8300"


def test_dockerfile_cmd_runs_the_entrypoint_we_tested() -> None:
    """Dockerfile 的 CMD 必须真的执行 scripts/run_api.py。

    上面几条测的都是 run_api.py 的行为；如果 CMD 换成了 uvicorn 命令行，
    那些断言就与线上跑的东西脱钩了（测 A 跑 B）。这条把它们焊在一起。
    """
    text = _read_repo_file("Dockerfile")
    assert "scripts/run_api.py" in text, "Dockerfile 的 CMD 不再执行 scripts/run_api.py"


def test_compose_publishes_the_same_port() -> None:
    """compose 的端口映射必须与默认 API_PORT 一致。

    不一致的症状是「进程绑对了，但映射到了另一个端口」—— 同样是宿主连不上，
    只是根因不同。两个数必须成对出现，所以放在同一个断言维度里。
    """
    text = _read_repo_file("docker-compose.yml")
    assert "8300:8300" in text, "compose 丢了 8300:8300 映射"
