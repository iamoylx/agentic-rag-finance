"""数字规则叶子模块：整个项目对「什么是数字、哪些数字要管」的唯一真相源。

语法：只依赖标准库 re。**刻意不 import 任何项目内模块**——保持为依赖树的叶子。
原理（为什么单独成文件）：
  数值安全有**两个消费者**：
    ① src/graph/nodes.py::check_hallucination —— 对「完整答案」判定，权威、离线、可单测
    ② src/safety/number_gate.py             —— 对「增量文本」判定，逐 token、实时
  两边必须用同一套正则 + 同一套过滤规则，否则闸门放行的内容权威校验可能判失败
  （或反向），用户就会看到「屏幕上明明没出数字，最后却说校验失败」这类诡异现象。
  而 ① 所在的 nodes.py 会连带 import 检索栈（torch / 向量库），② 如果在流式层做、
  不该背这个重量。所以把纯规则下沉到这里，让 `import src.safety.number_gate`
  不必付出加载 torch 的代价（单测因此能毫秒级跑完）。
"""
from __future__ import annotations

import re

# 数字字面量：整数部分允许千分位（半角/全角逗号），小数部分可选
NUM_RE = re.compile(r"\d[\d,，]*(?:\.\d+)?")


def norm_num(s: str) -> str:
    """数字归一化：去千分位/中文逗号并去掉尾部裸小数点，便于「答案数字 vs 原文数字」匹配。

    例："170,899,152,276.34" → "170899152276.34"
    """
    return s.replace(",", "").replace("，", "").rstrip(".")


def meaningful_numbers(text: str) -> set[str]:
    """抽出一段文本里「有意义的数字」（已归一化）。

    过滤规则：归一化后长度 < 2 的数字不算——这类基本是页码引用（"第5页"→"5"）
    或列表序号（"1." "2."），把它们纳入校验会导致大量误判。
    """
    nums = {norm_num(m.group()) for m in NUM_RE.finditer(text)}
    return {n for n in nums if len(n.replace(".", "")) >= 2}


def build_corpus_norm(docs) -> str:
    """把检索到的片段拼成归一化语料串——「答案数字必须能在里面找到」的比对母本。

    关键性质：它只依赖检索结果，**不依赖答案**。
    → 在 generate 开始之前就已完备，这正是「增量校验可行」的根据：
      判定一个数字是否编造，所需信息在它被生成出来之前就全都到手了。
    """
    return norm_num("\n".join(d.page_content for d in docs))


def missing_numbers(text: str, corpus_norm: str) -> list[str]:
    """text 中出现、但 corpus 中不存在的数字（即编造的数字）。空列表 = 通过。"""
    return [n for n in meaningful_numbers(text) if n not in corpus_norm]
