"""
服务启动脚本。

语法：sys.path 注入项目根 → uvicorn 以编程方式启动（省去命令行参数）。
用途：`python scripts/run_api.py` 一条命令起服务，默认 127.0.0.1:8300。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8300,
        log_level="info",
        # 冷启动要加载 BGE-M3 + Reranker，超时放宽，避免模型没加载完就被判失联
        timeout_keep_alive=120,
    )
