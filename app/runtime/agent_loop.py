"""Agent 循环（ReAct）：思考 → 行动 → 观察 → 反思。

每个子任务一段循环。LLM 输出结构化动作；工具结果以 scratch 行写回上下文，
同时写进事件流（崩溃恢复时由 EventStore.scratchpad_for_subtask 重建）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .actions import ActionParseError, parse_action
from .budget import Budget
from .context import build_context
from .events import EventStore
from .model_gateway import ModelProvider
from ..tools.registry import ToolRouter

SYSTEM_PROMPT = (
    "你是一个研究数字员工的执行引擎。每轮只输出一个 JSON 动作：\n"
    '{"action": "<动作名>", "args": {...}}\n'
    "动作参数表（args 的键必须按此填写，不要自创参数名）：\n"
    '- web_search: {"query": "搜索词"}\n'
    '- web_fetch: {"url": "完整 URL"}\n'
    '- write_artifact: {"type": "evidence", "title": "标题", "content": "正文", '
    '"refs": ["证据编号或 URL"]}\n'
    '- read_artifact: {"id": "工件 id（write_artifact 返回的 id 或 list_artifacts '
    '列出的 id，如 E-1）"}\n'
    '- list_artifacts: {} 或 {"type": "evidence"}\n'
    '- finish_subtask: {"summary": "本子任务一句话结论"}\n'
    "依据 SCRATCH 中已完成的步骤决定下一步；证据必须写入工件并带 refs。\n"
    "工具返回 ERROR 时修正参数或换工具继续，不要原样重试同一个失败调用。"
)


@dataclass
class SubtaskResult:
    summary: str
    gaps: list[str] = field(default_factory=list)
    steps: int = 0
    exhausted: bool = False


class AgentLoop:
    def __init__(self, llm: ModelProvider, tools: ToolRouter, events: EventStore,
                 budget: Budget, max_steps: int = 12, max_parse_retries: int = 2):
        self.llm = llm
        self.tools = tools
        self.events = events
        self.budget = budget
        self.max_steps = max_steps
        self.max_parse_retries = max_parse_retries

    def run_subtask(self, task_id: str, task: str, subtask: dict,
                    subtask_index: int, scratch: list[str]) -> SubtaskResult:
        parse_fails = 0
        for step in range(1, self.max_steps + 1):
            if not self.budget.can_call_llm():
                self.events.append(task_id, "subtask.paused", {
                    "subtask": subtask_index, "reason": "budget_llm_exhausted"})
                return SubtaskResult(summary=f"子任务 {subtask_index + 1} 因 LLM 预算耗尽暂停",
                                     steps=step - 1, exhausted=True)
            ctx = build_context("EXECUTE", task, subtask=subtask,
                                scratch=list(scratch))
            raw = self.llm.complete(SYSTEM_PROMPT, ctx)
            self.budget.consume_llm()
            self._record_llm_call(task_id, "EXECUTE", subtask_index, step, raw)

            try:
                action = parse_action(raw)
            except ActionParseError as e:
                parse_fails += 1
                if parse_fails > self.max_parse_retries:
                    self.events.append(task_id, "subtask.aborted", {
                        "subtask": subtask_index, "reason": "llm_output_unparseable"})
                    return SubtaskResult(
                        summary=f"子任务 {subtask_index + 1} 因 LLM 输出持续无法解析而中止",
                        gaps=["LLM 输出不可解析"], steps=step, exhausted=True)
                self.events.append(task_id, "llm.parse_error", {
                    "subtask": subtask_index, "step": step, "error": str(e)})
                scratch.append(f"PARSE_ERROR: {e}。请严格只输出一个 JSON 动作。")
                continue

            if action.name == "finish_subtask":
                self.events.append(task_id, "subtask.finished", {
                    "subtask": subtask_index, "summary": action.args.get("summary", ""),
                    "gaps": action.args.get("gaps", []), "steps": step})
                return SubtaskResult(summary=action.args.get("summary", ""),
                                     gaps=action.args.get("gaps", []), steps=step)

            if not self.budget.can_call_tool():
                self.events.append(task_id, "subtask.paused", {
                    "subtask": subtask_index, "reason": "budget_tool_exhausted"})
                return SubtaskResult(summary=f"子任务 {subtask_index + 1} 因工具预算耗尽暂停",
                                     steps=step, exhausted=True)
            result = self.tools.run(task_id, action.name, action.args, subtask=subtask_index)
            self.budget.consume_tool()
            self.events.append(task_id, "tool.called", {
                "stage": "EXECUTE", "subtask": subtask_index, "step": step,
                "tool": action.name, "args": action.args,
                "ok": result.ok, "error": result.error,
                "result": result.data, "scratch": result.scratch})
            scratch.append(result.scratch)
            # 观察（observation）已写回 scratch；失败时 LLM 下一轮可见错误并自行换路

        self.events.append(task_id, "subtask.max_steps", {"subtask": subtask_index})
        return SubtaskResult(summary=f"子任务 {subtask_index + 1} 达到步数上限收尾",
                            gaps=["步数上限"], steps=self.max_steps)

    def _record_llm_call(self, task_id: str, stage: str,
                         subtask: int | None, step: int, raw: str) -> None:
        """llm.called 事件：尽量带上 token/延迟/重试等调用元数据（provider 可选提供）。"""
        meta = getattr(self.llm, "last_meta", None)
        payload = {
            "stage": stage, "subtask": subtask, "step": step,
            "model": meta.model if meta else self.llm.name,
            "provider": meta.provider if meta else self.llm.name,
            "chars": len(raw),
        }
        if meta is not None:
            payload.update({"tokens_in": meta.tokens_in, "tokens_out": meta.tokens_out,
                            "latency_ms": meta.latency_ms, "retries": meta.retries})
            self.budget.consume_tokens(meta.tokens_total)
        self.events.append(task_id, "llm.called", payload)
