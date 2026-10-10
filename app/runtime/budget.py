"""预算控制：LLM 调用数 / 工具调用数 两类计数预算（M1 子集）。

软限 → 收尾模式：不再开新子任务，尽快产出含"未完成项"的报告。
硬限 → 由调用方（agent_loop）直接返回 exhausted 结果。
M2/M3 再补 token 与金额预算、时长预算。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Budget:
    """调用次数 + token 三类预算（M2 加入 token；M3 补金额/时长）。"""
    max_llm_calls: int = 200
    max_tool_calls: int = 400
    max_total_tokens: int = 0     # 0 = 不限
    llm_calls: int = 0
    tool_calls: int = 0
    tokens_used: int = 0

    def can_call_llm(self) -> bool:
        if self.llm_calls >= self.max_llm_calls:
            return False
        return self.max_total_tokens == 0 or self.tokens_used < self.max_total_tokens

    def can_call_tool(self) -> bool:
        return self.tool_calls < self.max_tool_calls

    def consume_llm(self) -> None:
        self.llm_calls += 1

    def consume_tool(self) -> None:
        self.tool_calls += 1

    def consume_tokens(self, n: int) -> None:
        self.tokens_used += n

    @property
    def exhausted(self) -> bool:
        tokens_hit = (self.max_total_tokens > 0
                      and self.tokens_used >= self.max_total_tokens)
        return (self.llm_calls >= self.max_llm_calls
                or self.tool_calls >= self.max_tool_calls
                or tokens_hit)
