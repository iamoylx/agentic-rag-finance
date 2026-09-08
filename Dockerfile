# Agentic RAG for Finance —— 生产部署镜像
# 语法：多阶段没有收益（模型在卷里不进镜像），单阶段 + 卷挂载是最小方案。
# 原理：BGE-M3 / Reranker / 年报数据全部走卷挂载（models/ 和 data/），镜像只含代码与依赖
#      ——换模型不用重建镜像，重建镜像不用重新下载 2GB 权重。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBED_DEVICE=cpu

WORKDIR /app

# 先装依赖再拷代码：代码改动不触发依赖重装（层缓存）
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY scripts/run_api.py ./scripts/run_api.py
COPY web/ ./web/

EXPOSE 8300
CMD ["python", "scripts/run_api.py"]
