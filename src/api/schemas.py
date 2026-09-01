"""
API 请求 / 响应模型。

语法：pydantic v2 BaseModel，自动做类型校验与序列化。
用途：把 HTTP 层的输入输出契约集中在一个文件，路由函数只做编排不做校验。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/chat 的请求体。session_id 对应 LangGraph checkpointer 的 thread_id。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str = Field("default", max_length=64, description="会话 ID，多轮追问靠它串起来")


class IngestRequest(BaseModel):
    """POST /api/ingest 的请求体。path 指向项目内（或本机任意）PDF 文件。"""

    path: str = Field(..., description="PDF 文件路径")
    company: str = Field("", description="公司名，注入每个分块的上下文前缀")
    year: str = Field("", description="报告年份，同上")


class Citation(BaseModel):
    """引用徽章：前端据此渲染 #pX 并跳转原文页。"""

    source: str
    page: int
    snippet: str


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    llm_provider: str
    doc_count: int
    chunk_count: int
