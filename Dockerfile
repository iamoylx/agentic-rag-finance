# Agentic RAG for Finance —— 生产部署镜像
#
# 为什么仍然「模型不进镜像」：
#   BGE-M3 / Reranker 权重（约 2 GB）与年报数据全部走卷挂载（models/ 和 data/），
#   镜像只含代码与依赖——换模型不用重建镜像，重建镜像不用重新下载权重。
#
# 阶段顺序很要紧：**生产必须是最后一个阶段**。`docker compose` 与 `docker build .`
# 不带 --target 时构建的是最后一个阶段；若把 CI 测试阶段放最后，生产镜像会变成测试镜像。
#
#   docker build .                      → runtime（生产，默认）
#   docker build --target test -t t .   → test（CI 在容器内跑 pytest）
#
# 语法：先装依赖再拷代码，代码改动不触发依赖重装（层缓存命中率最高）。

# ---------- 阶段 1：依赖（生产与测试共用）----------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBED_DEVICE=cpu

WORKDIR /app

# 两个文件都要拷：requirements-docker.txt 里有 `-r requirements.txt`，
# 只拷前者会让 pip 直接报 "Could not open requirements file: requirements.txt"
# （这个 bug 曾真实存在——该镜像从未被构建验证过）。
COPY requirements.txt requirements-docker.txt ./

# torch 单独装，且用 --index-url（**不是** --extra-index-url）：
# 只有单索引才能确定拿到 CPU 版；extra-index 会让 pip 在 PyPI 与 CPU 索引之间按版本号挑，
# 结果不确定（PyPI 的 linux torch 带 CUDA，镜像会凭空胖 2-3 GB）。
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements-docker.txt

# ---------- 阶段 2：CI 测试（不进生产镜像）----------
# 必须同时拷 src/ **和** web/，只拷 tests/ 会在收集阶段就炸：
#   - 缺 src/  → `ModuleNotFoundError: No module named 'src'`（4 个用例文件全部 ERROR）
#   - 缺 web/  → src/api/app.py 在导入期执行 `StaticFiles(directory=PROJECT_ROOT/"web")`，
#                目录不存在时 Starlette 直接抛 RuntimeError
# 这个坑在第一条 GitHub Actions 里真实红过一次（镜像能建、容器内 pytest 挂）。
#
# 还要拷 scripts/ 与 evals/：`tests/test_output_encoding.py` 的 AST 闸门会扫
# SCAN_DIRS = (src, scripts, evals, tests) 四个目录，而 `test_all_scan_dirs_present`
# 明确断言这四个目录都必须在——少一个就报错，不让闸门**静默缩小扫描面**。
# （这条用例加上之后，立刻反过来抓到了本 Dockerfile 自己：原来只拷了 src/web/tests。）
#
# 顺序上代码在前、tests 在后：tests 改动最频繁，放最后一层能让上面几层缓存继续命中。
FROM base AS test
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY evals/ ./evals/
COPY web/ ./web/
COPY pytest.ini ./
COPY tests/ ./tests/
CMD ["python", "-m", "pytest", "-q"]

# ---------- 阶段 3：生产镜像（默认 target）----------
FROM base AS runtime
COPY src/ ./src/
COPY scripts/run_api.py ./scripts/run_api.py
COPY web/ ./web/

# 刻意**不**把 evals/ 放进生产镜像（T2 决策，2026-10-20）：
#   ① 本项目的 evals 全部需要「真索引 + 真 LLM」才能跑，容器里没有卷挂载就跑不了；
#   ② 它们是**开发期工具**（消融扫描、金标构建），不是运行时资源——
#      与 P2 不同，P1 的 src/ 在运行期不读 evals/ 下任何文件。
# 要在容器里跑评测，用 `docker run -v $PWD/evals:/app/evals -v $PWD/data:/app/data ...`
# 临时挂进去，别让它们常驻镜像。

EXPOSE 8300
CMD ["python", "scripts/run_api.py"]
