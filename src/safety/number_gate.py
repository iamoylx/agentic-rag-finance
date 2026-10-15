"""流式数值显示闸门（Numeric Hold-Back Gate）：只把「已校验过的数字」放上屏幕。

语法：纯同步类，接口只有三个——feed(chunk) -> str、flush() -> str、violated 标志。
      不依赖 FastAPI / asyncio / langgraph，因此可以毫秒级单测。
原理：
  check_hallucination 是**纯规则**判定（数字归一化后查语料子串），而语料在 generate
  开始前就已确定 → 「校验」不必等整段答案结束，每个数字一凑齐就能即时判定。
  于是把校验时机从「**生成之后**」提前到「**生成之中**」：
    扣住正在拼装的数字 → 凑齐即判定 → 通过才放行。
  屏幕上因此永远不出现未经验证的数字，而正文（无数字的部分）照常逐字流式输出。

  为什么不是「整段扣留到校验完再放」：那会让首字延迟 = 整次生成耗时，
  等于牺牲掉流式体验（见 METRICS / 手册第 7 章的权衡分析）。

与权威校验的关系：本闸门**不取代** src/graph/nodes.py::check_hallucination。
  图里的节点仍是唯一权威（它决定 finalize / 重生成 / 降级），闸门只负责「屏幕」
  这一层的显示时机。两者共用 src/safety/number_rules.py，判定必然一致。
"""
from __future__ import annotations

import re

from src.safety.number_rules import missing_numbers

# 缓冲区末尾若匹配到「数字前缀」，说明它可能还没拼完 → 一律扣住不放行。
# 必须容忍千分位与小数点：模型会把 "170,899,152,276.34" 拆成多个 token
# （如 "170,8" / "99,15" / "2,276.34"），中途任何前缀都是「合法的数字前缀」，
# 只有等到后面跟了非数字字符，才能确认这个数字已经完整、可以判定。
_TRAILING_NUM_RE = re.compile(r"\d[\d,，]*(?:\.\d+)?$")


class NumericHoldBackGate:
    """扣留「正在拼装中的数字」，凑齐后即刻校验，通过才放行。

    用法（在 SSE 生成器里）：
        gate = NumericHoldBackGate(corpus_norm)
        for chunk in tokens:
            out = gate.feed(chunk)
            if out:
                yield sse_token(out)
        tail = gate.flush()          # 流结束必须 flush，否则末尾数字永远不放行
    """

    def __init__(self, corpus_norm: str) -> None:
        self._corpus = corpus_norm
        self._buf = ""           # 被扣留的「可能还没拼完」的尾部
        self.violated = False    # 是否已发现编造数字（一旦为真，本闸门不再放行任何内容）

    # ---------- 增量入口 ----------
    def feed(self, chunk: str) -> str:
        """喂入一个增量 token，返回「此刻可安全上屏」的文本（可能为空串）。"""
        if self.violated:
            return ""
        self._buf += chunk

        # 尾部若是数字前缀 → 扣住；其前的部分才是「已定型」的文本
        m = _TRAILING_NUM_RE.search(self._buf)
        cut = m.start() if m else len(self._buf)
        safe, self._buf = self._buf[:cut], self._buf[cut:]

        # 只有通过判定才放行——这是「屏幕上不出未校验数字」这条不变式的唯一保证点
        if safe and missing_numbers(safe, self._corpus):
            self.violated = True
            return ""
        return safe

    # ---------- 收尾入口 ----------
    def flush(self) -> str:
        """流结束：把仍被扣留的尾部放行（放行前同样要判定）。

        必须调用：若答案恰好以数字结尾（"...为68.64"），不 flush 就会永久扣住，
        屏幕少一截；而 flush 里做判定保证「即便在最末尾也不放未校验的数字」。
        """
        if self.violated:
            self._buf = ""
            return ""
        out, self._buf = self._buf, ""
        if out and missing_numbers(out, self._corpus):
            self.violated = True
            return ""
        return out
