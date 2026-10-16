"""
全局配置模块。

用途：项目唯一的配置入口。所有路径、密钥、超参都从这里读，禁止在业务代码里散落 os.getenv / 硬编码路径。
原理：
  - pydantic-settings 的 BaseSettings 会自动从「环境变量」和「.env 文件」读取字段，
    字段名 deepseek_api_key 会自动匹配环境变量 DEEPSEEK_API_KEY（大小写不敏感）。
  - 所有 *_dir / *_path 都基于 PROJECT_ROOT 推导，保证任何产物都落在项目目录内，
    满足「禁止在项目空间之外写入」的硬约束。
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：src/core/config.py -> parents[0]=core, [1]=src, [2]=项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # env_file 不存在时 pydantic-settings 会静默跳过，因此不强制要求 .env
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- LLM 提供方切换 ----------
    # agnes：免费额度，OpenAI 协议兼容（当前主力）
    # deepseek：备用——无自动切换逻辑，改 llm_provider 字段手动切换
    llm_provider: str = "agnes"            # agnes | deepseek

    # Agnes（多模态平台，OpenAI 兼容；AGNES_API_KEY 已配置在用户级环境变量）
    agnes_api_key: str = ""
    agnes_base_url: str = "https://apihub.agnes-ai.com/v1"
    agnes_model: str = "agnes-2.0-flash"   # 文本模型，256K 上下文，支持工具调用

    # DeepSeek（OpenAI 协议兼容；DEEPSEEK_API_KEY 已配置在用户级环境变量）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"       # deepseek 专用模型名；agnes 用 agnes_model
    llm_temperature: float = 0.0           # RAG 场景要确定性，温度设 0
    llm_max_tokens: int = 4096             # 注意：agnes 需 ≥256，过小会整段返回空
    llm_timeout: int = 120

    # DeepSeek-V4 起「思考模式」默认开启（effort=high），对本项目有两个副作用：
    #   ① 思维链的 token 计入输出计费 → 成本可高出数倍；
    #   ② 输出预算会被思考消耗掉，max_tokens 偏小时 content 可能为空
    #      —— 这正是 Agnes 那个坑的同款（见 llm.py 模块 docstring）。
    # 本项目是「照原文答数」的 RAG，不需要长链推理，故默认关闭；
    # 想对比思考模式对准确率的贡献时改这一项即可（可选 enabled | disabled）。
    deepseek_thinking: str = "disabled"

    # ---------- 联网降级（Corrective RAG 的兜底检索）----------
    tavily_api_key: str = ""
    web_search_max_results: int = 3

    # ---------- 本地模型：BGE-M3（向量化） / BGE-Reranker（精排）----------
    # 路径锚定项目内 models/ 目录，不使用 HF 默认缓存（默认缓存会写 C 盘）
    embed_model_path: str = str(PROJECT_ROOT / "models" / "bge-m3")
    rerank_model_path: str = str(PROJECT_ROOT / "models" / "bge-reranker-v2-m3")
    embed_device: str = "cuda"             # cuda / cpu；显存不足时改 cpu
    embed_batch_size: int = 12             # 4070 8GB 建议 8-16，再大易 OOM
    embed_max_length: int = 1024           # BGE-M3 支持最长 8192，财报分块 1024 够用
    use_fp16: bool = True                  # fp16 显存减半、速度翻倍，精度损失可忽略

    # ---------- 检索参数 ----------
    chunk_size: int = 512                  # 中文约 1 字 ≈ 1 token
    chunk_overlap: int = 64
    top_k_retrieve: int = 20               # 粗召回数量（送入精排前）
    top_k_final: int = 5                   # 最终进入 Prompt 的块数（对应 Recall@5）
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60                        # RRF 融合常数，SIGIR 2009 原论文取 60
    # 加权 RRF：消融实测（26 条真实年报评测集）等权 RRF 会被 BM25 长尾噪声稀释，
    # dense 命中被挤出融合 top-5（Recall@5 0.923→0.692）；提高 dense 权重后恢复
    rrf_dense_weight: float = 0.7
    rrf_bm25_weight: float = 0.3

    # ---------- Agent 自纠正循环上限（防死循环烧 Token）----------
    max_rewrite: int = 2                   # 查询改写最多重试次数
    max_retrieve_rounds: int = 3           # 最多检索轮次
    max_regenerate: int = 2                # 幻觉校验不通过时最多重新生成次数

    # ---------- 缓存 / 存储 ----------
    # 留空则自动使用 fakeredis（进程内内存版，接口与 redis-py 完全一致）
    redis_url: str = ""
    mysql_url: str = ""                    # 例：mysql+pymysql://user:pass@127.0.0.1:3306/rag

    # ---------- 路径（全部项目内）----------
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw"

    @property
    def parsed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "parsed"

    @property
    def chroma_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "chroma"

    @property
    def dict_path(self) -> Path:
        return PROJECT_ROOT / "data" / "dict" / "finance_terms.txt"

    def ensure_dirs(self) -> None:
        """创建所有必需的目录。用途：启动时调用一次，避免各处 mkdir。"""
        for d in [self.data_dir, self.raw_dir, self.parsed_dir, self.chroma_dir,
                  self.dict_path.parent, PROJECT_ROOT / "logs"]:
            d.mkdir(parents=True, exist_ok=True)


# 全局单例：其他模块 `from src.core.config import settings`
settings = Settings()
