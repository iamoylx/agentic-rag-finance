"""
LLM 客户端工厂（多提供方）。

用途：全项目唯一的 LLM 实例来源。通过 settings.llm_provider 在 Agnes / DeepSeek 间切换，
     上层代码完全无感。
原理：
  - 两家都实现了 OpenAI 协议，因此统一用 langchain_openai.ChatOpenAI，
    仅 base_url / api_key / model 三个参数不同。
  - lru_cache 保证同参数只创建一次实例：复用 httpx 连接池，避免每次调用重建连接。
  - 注意坑：Agnes 对 max_tokens 敏感，过小（如 16）会返回空 content 而不报错，
    因为输出预算被内部思考消耗殆尽。因此 get_cheap_llm 最低给 512。
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.core.config import settings


def _resolve_provider() -> tuple[str, str, str]:
    """根据 llm_provider 解析出 (api_key, base_url, model)。

    用途：集中处理提供方切换逻辑，报错信息直接给出修复指引。
    """
    p = settings.llm_provider.lower()
    if p == "agnes":
        if not settings.agnes_api_key:
            raise RuntimeError("llm_provider=agnes 但未读取到 AGNES_API_KEY（用户级环境变量）")
        return settings.agnes_api_key, settings.agnes_base_url, settings.agnes_model
    if p == "deepseek":
        if not settings.deepseek_api_key:
            raise RuntimeError("llm_provider=deepseek 但未读取到 DEEPSEEK_API_KEY")
        return settings.deepseek_api_key, settings.deepseek_base_url, settings.llm_model
    raise ValueError(f"未知 llm_provider: {settings.llm_provider}（可选 agnes / deepseek）")


@lru_cache(maxsize=8)
def get_llm(temperature: float | None = None, max_tokens: int | None = None) -> ChatOpenAI:
    """获取 LLM 实例。

    用途：供 graph / retrieval / evals 各处调用。
    参数：temperature 覆盖默认（默认 0，RAG 要确定性输出）；max_tokens 同理。
    """
    api_key, base_url, model = _resolve_provider()
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
        timeout=settings.llm_timeout,
        max_retries=2,
        # SSE 打字机效果的前提：langgraph stream_mode="messages" 依赖
        # on_llm_new_token 回调，而非流式调用只会一次性吐完整消息
        streaming=True,
    )


def get_cheap_llm() -> ChatOpenAI:
    """获取轻量 LLM（路由判断、相关性分级等短输出分类任务）。

    用途：Adaptive RAG 的查询路由、Corrective RAG 的相关性打分。
    原理：这类任务是短输出分类，1024 token 足够且显著省成本；
          对 Agnes 而言 max_tokens 不能太小（见模块 docstring 的坑）。
    """
    return get_llm(temperature=0.0, max_tokens=1024)
