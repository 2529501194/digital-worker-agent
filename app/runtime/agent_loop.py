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
    "参数必须写在 args 对象内（不要写在顶层）；query/url 等必填参数不得留空。\n"
    "依据 SCRATCH 中已完成的步骤决定下一步；搜到材料后尽快用 write_artifact 把证据落盘"
    "（带 refs），不要一直搜索。证据必须写入工件并带 refs。\n"
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
        import json as _json
        parse_fails = 0
        failed_sigs: dict[str, int] = {}   # 同参失败调用计数（退化循环护栏）
        wrote_evidence = False
        soft_landed = False
        for step in range(1, self.max_steps + 1):
            if not self.budget.can_call_llm():
                self.events.append(task_id, "subtask.paused", {
                    "subtask": subtask_index, "reason": "budget_llm_exhausted"})
                return SubtaskResult(summary=f"子任务 {subtask_index + 1} 因 LLM 预算耗尽暂停",
                                     steps=step - 1, exhausted=True)
            remaining = self.max_steps - step + 1
            if step >= self.max_steps - 1 and (remaining == 1 or not soft_landed):
                # 软着陆：剩最后 2 步提示收口（先落证据再 finish，不开新搜索）；
                # 最后 1 步即使已提示过也要再注入强制收口（模型可能无视弱提示）。
                # t-59d18668：三个子任务全部 max_steps 收尾，writer 拿不到
                # 子任务结论，报告质量受损；给模型一个明确的收口时机。
                if remaining > 1:
                    scratch.append(
                        f"SOFT_LANDING: 步数即将用尽（剩 {remaining} 步）。"
                        "请立即：1) 用 write_artifact 把已收集的材料落盘为证据；"
                        "2) 用 finish_subtask 提交一句话结论收尾；不要再发起新的搜索/抓取。")
                else:
                    # t-dabc6b7b：剩 1 步时模型仍可能发起 read/fetch 而撞顶，
                    # 收尾语退化成模板句——最后一步强制收口
                    scratch.append(
                        "SOFT_LANDING: 这是最后 1 步，必须调用 finish_subtask 收尾"
                        "（提交一句话结论 + gaps）；若证据未落盘可先 write_artifact，"
                        "之后立即 finish_subtask，不要再调用其他工具。")
                soft_landed = True
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
            if action.name == "write_artifact" and result.ok:
                wrote_evidence = True
            if not result.ok:
                sig = (action.name, _json.dumps(action.args, sort_keys=True,
                                               ensure_ascii=False, default=str))
                failed_sigs[sig] = failed_sigs.get(sig, 0) + 1
                if failed_sigs[sig] >= 2:
                    # 退化循环护栏：同一失败调用第二次出现即强提示换路
                    scratch.append(
                        "REPEAT_FAIL: 你已用完全相同的参数失败 "
                        f"{failed_sigs[sig]} 次（{action.name} args={action.args}）。"
                        "检查必填参数是否真的写进了 args 对象（对照参数表），"
                        "换搜索词/换工具/先 list_artifacts 查 id；材料足够时立即用 "
                        "write_artifact 落证据，不要再重复这个失败调用。")
            # 观察（observation）已写回 scratch；失败时 LLM 下一轮可见错误并自行换路

        gaps = ["步数上限"]
        summary = f"子任务 {subtask_index + 1} 达到步数上限收尾"
        if not wrote_evidence:
            gaps.append("未写入证据")
            summary += "，且全程未写入任何证据工件（write_artifact）"
        # summary/gaps 持久化进事件：writer（含崩溃恢复后）可重建子任务结论
        self.events.append(task_id, "subtask.max_steps",
                          {"subtask": subtask_index, "summary": summary,
                           "gaps": gaps})
        return SubtaskResult(summary=summary, gaps=gaps, steps=self.max_steps)

    def _record_llm_call(self, task_id: str, stage: str,
                         subtask: int | None, step: int, raw: str) -> None:
        """llm.called 事件：尽量带上 token/延迟/重试等调用元数据（provider 可选提供）。"""
        meta = getattr(self.llm, "last_meta", None)
        payload = {
            "stage": stage, "subtask": subtask, "step": step,
            "model": meta.model if meta else self.llm.name,
            "provider": meta.provider if meta else self.llm.name,
            "chars": len(raw),
            # 原文前 300 字符：排障时可直接看到模型实际输出（参数写顶层/留空等
            # 静默丢参问题在事件流里不可见，是 t-a58c3f76 复盘的关键缺口）
            "raw_head": raw[:300],
        }
        if meta is not None:
            payload.update({"tokens_in": meta.tokens_in, "tokens_out": meta.tokens_out,
                            "latency_ms": meta.latency_ms, "retries": meta.retries})
            self.budget.consume_tokens(meta.tokens_total)
        self.events.append(task_id, "llm.called", payload)
