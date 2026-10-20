"""
模型完整性验证：真实加载并跑一次推理。

用途：光比对文件大小不够，必须能加载、能编码、维度正确才算数。
原理：
  - BGE-M3 是 XLM-RoBERTa 架构，transformers 原生 AutoModel 即可加载。
  - 句向量取 CLS 位（config.json 里 pooling_mode_cls_token=true），再 L2 归一化。
  - Reranker 是 CrossEncoder：把 (query, doc) 拼成一个输入，输出单一相关性分数。

运行：python scripts/verify_models.py
      python scripts/verify_models.py > verify.log 2>&1   # 重定向到文件也受支持（写 UTF-8）

★ 输出编码（2026-10-20 修）：Windows 控制台默认 cp936(GBK)，**能编码中文、不能编码 emoji**
  （`UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'`）。原先脚本最后一行打
  「✅ 全部通过」，于是**验证明明跑完、最后一步却崩在打印上**，退出码变成非零——
  看起来像模型坏了，其实是 IO 编码问题。修法两条并存：
    ① 状态标记改 ASCII（`[OK]`/`[FAIL]`/`[WARN]`），中文正文保留（GBK 表示中文没问题）；
    ② `_setup_stdio()` 兜底：终端只降级 errors，重定向到文件则统一 UTF-8。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from src.core.config import settings  # noqa: E402

SEP = "=" * 70

# 状态标记用 ASCII：GBK 控制台编码不了 ✅/❌，而这些恰恰出现在「结论行」上——
# 一旦崩在这里，人看到的是 traceback，会误判成模型/推理出问题。
OK, FAIL, WARN = "[OK]", "[FAIL]", "[WARN]"


def _setup_stdio() -> None:
    """把 stdout/stderr 的编码风险一次性处理掉。

    两种情形分开处理，因为正确解不一样：
      · **终端**：不改编码。cp936 下中文本来就是对的，硬改成 UTF-8 反而会让控制台
        按 GBK 解读 UTF-8 字节 → 满屏乱码。只把 `errors` 降级，让漏网的不可编码字符
        变成 `?`，而不是抛 `UnicodeEncodeError` 把整个验证中断。
      · **重定向到文件/管道**：统一 UTF-8。日志要能跨机器、跨编辑器读，跟随 locale 的
        cp936 会在别人那边变成乱码。

    `reconfigure` 只有 Python 3.7+ 的 TextIOWrapper 有；pytest 的 capture、`io.StringIO`、
    被别的库包过的流都没有——用 getattr 探测，没有就跳过（也不该硬改）。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 流已关闭/已被接管：宁可不改编码，也不该因为「日志格式」中断验证
            continue


def _cls_pooling(last_hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    """CLS 池化 + L2 归一化。

    用途：把 token 级隐状态转成句向量。
    原理：BGE 系列训练时以 [CLS] 表征整句；归一化后用点积即余弦相似度，便于检索。
    """
    vecs = last_hidden[:, 0]
    return torch.nn.functional.normalize(vecs, p=2, dim=1)


def check_embed() -> bool:
    print(SEP)
    print("验证 1/2  BGE-M3（向量化）")
    path = Path(settings.embed_model_path)
    if not (path / "pytorch_model.bin").exists() and not (path / "model.safetensors").exists():
        print(f"  {FAIL} 权重文件缺失: {path}")
        return False

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModel.from_pretrained(str(path))
    model.eval()
    print(f"  加载完成 {time.time() - t0:.1f}s")

    dev = settings.embed_device if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        model = model.cuda().half() if settings.use_fp16 else model.cuda()
    print(f"  推理设备: {dev}{' (fp16)' if dev == 'cuda' and settings.use_fp16 else ''}")

    sents = [
        "公司2025年实现营业收入123.45亿元，同比增长12.36%",
        "报告期内研发投入占营业收入的比例为15.20%",
        "今天天气不错",  # 干扰句，用于验证语义区分度
    ]
    with torch.no_grad():
        enc = tok(sents, padding=True, truncation=True,
                  max_length=settings.embed_max_length, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        out = model(**enc)
        vecs = _cls_pooling(out.last_hidden_state, enc["attention_mask"])

    dim = vecs.shape[1]
    # 余弦相似度：句1与句2都是财报语境应更接近，与天气句应显著更远
    sim_12 = float(torch.dot(vecs[0], vecs[1]))
    sim_13 = float(torch.dot(vecs[0], vecs[2]))
    print(f"  向量维度: {dim}  （BGE-M3 期望 1024）")
    print(f"  财报句 vs 财报句 余弦相似度: {sim_12:.4f}")
    print(f"  财报句 vs 天气句 余弦相似度: {sim_13:.4f}")
    ok = dim == 1024 and sim_12 > sim_13
    print(f"  结论: {'[OK] 维度正确且语义区分正常' if ok else '[FAIL] 维度异常'}")
    return ok


def check_rerank() -> bool:
    print(SEP)
    print("验证 2/2  BGE-Reranker-v2-m3（精排）")
    path = Path(settings.rerank_model_path)
    if not (path / "model.safetensors").exists() and not (path / "pytorch_model.bin").exists():
        print(f"  {FAIL} 权重文件缺失: {path}")
        return False

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(path))
    # 关键：Reranker 是序列分类模型（num_labels=1），必须用 SequenceClassification 类加载，
    # 用 AutoModel 加载会缺分类头，输出里没有 logits
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.eval()
    print(f"  加载完成 {time.time() - t0:.1f}s")

    dev = settings.embed_device if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        model = model.cuda().half() if settings.use_fp16 else model.cuda()

    query = "公司2025年的营业收入是多少"
    cands = [
        "公司2025年实现营业收入123.45亿元，同比增长12.36%",   # 高度相关
        "报告期内研发投入合计18.76亿元",                        # 弱相关
        "董事会保证本报告内容真实、准确、完整",                  # 不相关
    ]
    scores: list[float] = []
    with torch.no_grad():
        for c in cands:
            enc = tok(query, c, padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            s = model(**enc).logits.view(-1).float().item()
            scores.append(s)

    for c, s in zip(cands, scores):
        print(f"  score={s:8.4f}  {c[:34]}")
    ok = scores[0] > scores[1] > scores[2]
    print(f"  结论: {'[OK] 相关性排序正确' if ok else '[FAIL] 排序异常'}")
    return ok


def main() -> int:
    _setup_stdio()          # 必须在任何 print 之前
    print(f"CUDA 可用: {torch.cuda.is_available()} | "
          f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    r1 = check_embed()
    r2 = check_rerank()
    print(SEP)
    print(f"模型验证总结果: {'[OK] 全部通过' if (r1 and r2) else '[FAIL] 有失败项'}")
    print(SEP)
    return 0 if (r1 and r2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
