"""
服务启动脚本。

语法：sys.path 注入项目根 → uvicorn 以编程方式启动（省去命令行参数）。
用途：`python scripts/run_api.py` 一条命令起服务，默认 127.0.0.1:8300。

容器里必须把 host 改成 0.0.0.0（Dockerfile 的 runtime 阶段已设 API_HOST=0.0.0.0）：
绑 127.0.0.1 时进程只监听容器内的 loopback，compose 里写 `8300:8300` 也送不进来，
而 healthcheck 在**容器内**跑 localhost 永远是通的——症状就是「docker ps 显示 healthy、
宿主机一个字节都收不到」。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8300")),
        log_level="info",
        # 冷启动要加载 BGE-M3 + Reranker，超时放宽，避免模型没加载完就被判失联
        timeout_keep_alive=120,
    )
