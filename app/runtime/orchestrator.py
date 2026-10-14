"""任务编排器：状态机 + 计划驱动 + 断点恢复。

    PLAN → EXECUTE（逐子任务 Agent 循环）→ WRITE → DONE
          ↑________________ 预算耗尽 → 收尾（部分报告 + paused）

恢复语义：每个子任务完成后落 checkpoint（cursor = 下一个子任务下标）。
进程崩溃后，新的 Orchestrator 用同一 EventStore 启动 → load_checkpoint
→ 从 cursor 继续；已完成子任务不重做，scratchpad 由事件流重建。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .agent_loop import AgentLoop, SYSTEM_PROMPT
from .budget import Budget
from .context import build_context
from .events import EventStore
from .model_gateway import ModelProvider
from ..tools.artifact_store import ArtifactStore
from ..tools.registry import ToolRouter

PLAN_PROMPT = (
    "根据研究任务生成结构化研究计划，只输出 JSON（不要 markdown 围栏、不要解释）：\n"
    '{"subtasks": [{"title": ..., "focus": ...}], "compare_dimensions": ["...", "..."]}\n'
    "要求：\n"
    "1) 每个子任务对应一个具体研究对象（与研究主题相关的具体产品/品牌/事件/组织），"
    "title 与 focus 必须包含具体对象名，禁止“收集列表/分析某类技术”这类没有具体对象的泛任务；\n"
    "2) focus 用于后续搜索资料，必须包含对象名 + 关键概念；若搜索需要其他语言，附该语言关键词（如英文）；\n"
    "3) compare_dimensions 是报告“对比分析”章节使用的对比维度（2-4 个），"
    "必须由研究主题本身推导（例如手机市场主题：市场份额 / 价格带 / 目标人群 / 核心卖点；"
    "Agent 架构主题：任务规划 / 工具使用 / 上下文管理），禁止输出与研究主题无关的维度。"
)

WRITE_PROMPT = (
    "你是研究报告撰写引擎。基于上下文中的研究计划与全部证据工件，"
    "直接输出完整的 Markdown 研究报告（不要 JSON、不要 markdown 围栏）：\n"
    "1) 概述；2) 每个研究对象的独立小节，论断必须引用证据编号与 URL；"
    "3) 对比分析（维度：{dims}）；4) 参考文献列表。\n"
    "硬性要求：参考文献只能使用“证据工件”里出现过的 URL，严禁编造或臆测链接；"
    "某对象没有证据时就如实标注“证据不足”，不得用占位 URL 凑数。"
)


def _extract_json(raw: str):
    """从 LLM 输出中提取第一个可解析 JSON 对象（容忍 markdown 围栏与前后杂文）。

    括号平衡提取（actions.first_json_object）：模型输出多个 JSON / 后随杂文时
    只取第一个完整对象——旧贪婪正则 r"\\{.*\\}" 会跨对象过度匹配而失败。
    """
    from .actions import first_json_object
    try:
        return first_json_object(raw)
    except ValueError as e:
        raise ValueError(f"LLM 输出中无有效 JSON（{e}）") from e


class _SimulatedCrash(Exception):
    """测试用：模拟进程在子任务边界崩溃（checkpoint 已在上一步落盘）。"""


class BudgetExhausted(Exception):
    """预算耗尽（PLAN/WRITE 阶段调用 LLM 前检查）。"""


@dataclass
class RunOutcome:
    status: str          # done | paused
    report_path: str | None
    completed_subtasks: int
    total_subtasks: int
    gaps: list[str]


class Orchestrator:
    def __init__(self, task_id: str, task_prompt: str,
                 llm: ModelProvider, tools: ToolRouter, events: EventStore,
                 artifacts: ArtifactStore, budget: Budget,
                 report_dir: str | Path,
                 plan_limit: int | None = None):
        self.task_id = task_id
        self.task = task_prompt
        self.llm = llm
        self.tools = tools
        self.events = events
        self.artifacts = artifacts
        self.budget = budget
        self.report_dir = Path(report_dir)
        self.plan_limit = plan_limit
        self.loop = AgentLoop(llm, tools, events, budget)

    def _llm_call(self, stage: str, system: str, user: str) -> str:
        """PLAN/WRITE 阶段的 LLM 调用：统一记录事件（含 token/延迟）并消耗预算。"""
        if not self.budget.can_call_llm():
            raise BudgetExhausted("LLM 预算耗尽，无法继续")
        raw = self.llm.complete(system, user)
        self.budget.consume_llm()
        meta = getattr(self.llm, "last_meta", None)
        payload = {"stage": stage, "model": meta.model if meta else self.llm.name,
                   "provider": meta.provider if meta else self.llm.name,
                   "chars": len(raw)}
        if meta is not None:
            payload.update({"tokens_in": meta.tokens_in, "tokens_out": meta.tokens_out,
                            "latency_ms": meta.latency_ms, "retries": meta.retries})
            self.budget.consume_tokens(meta.tokens_total)
        self.events.append(self.task_id, "llm.called", payload)
        return raw

    # ---------------------------------------------------------------- run
    def run(self, stop_after_subtasks: int | None = None) -> RunOutcome:
        """stop_after_subtasks 仅供测试模拟"崩溃"（跑到第 N 个子任务即中断）。"""
        if self.events.latest_seq(self.task_id) == 0:
            self.events.append(self.task_id, "task.created", {"prompt": self.task})
        try:
            return self._run_impl(stop_after_subtasks)
        except BudgetExhausted:
            self.events.append(self.task_id, "task.paused",
                               {"reason": "budget_exhausted_before_stage"})
            return RunOutcome(status="paused", report_path=None,
                              completed_subtasks=0, total_subtasks=0,
                              gaps=["预算耗尽，任务暂停（可 resume）"])

    def _run_impl(self, stop_after_subtasks: int | None) -> RunOutcome:
        cp = self.events.load_checkpoint(self.task_id)
        if cp is None or cp.stage == "PLAN":
            plan = self._plan()
            cursor = 0
        else:
            plan = self._load_plan()
            cursor = cp.subtask_cursor
            self.events.append(self.task_id, "task.resumed",
                               {"stage": cp.stage, "from_cursor": cursor})

        if cursor < len(plan):
            try:
                outcome = self._execute(plan, cursor, stop_after_subtasks)
            except _SimulatedCrash:
                return None  # 模拟进程死亡：无正常返回，等待 resume
            if outcome is not None:
                return outcome  # 预算耗尽 → 收尾

        return self._write()

    # ---------------------------------------------------------------- plan
    def _plan(self) -> list[dict]:
        prompt = PLAN_PROMPT
        if self.plan_limit:
            prompt += f"\n注意：子任务数量不超过 {self.plan_limit} 个。"
        raw = self._llm_call("PLAN", prompt, build_context("PLAN", self.task,
                                                           plan_limit=self.plan_limit))
        plan_data = _extract_json(raw)
        plan = plan_data.get("subtasks") or []
        if self.plan_limit and len(plan) > self.plan_limit:
            plan = plan[:self.plan_limit]  # 真实模型可能不严格遵守，硬截断兜底
        dims = [str(d).strip() for d in (plan_data.get("compare_dimensions") or [])
                if str(d).strip()][:4]
        self.events.append(self.task_id, "plan.created",
                           {"subtasks": plan, "compare_dimensions": dims})
        self.artifacts.write(self.task_id, "plan", "研究计划",
                             json.dumps({"subtasks": plan, "compare_dimensions": dims},
                                        ensure_ascii=False, indent=2))
        self.events.save_checkpoint(self.task_id, stage="EXECUTE", subtask_cursor=0)
        return plan

    def _load_plan(self) -> list[dict]:
        # 取**最后一次** plan.created：无检查点重跑（如 PLAN 阶段崩溃后重新运行）
        # 会生成新计划，旧计划作废，恢复必须以最新计划为准
        plan = None
        for _seq, type_, payload in self.events.replay(self.task_id):
            if type_ == "plan.created":
                plan = payload["subtasks"]
        if plan is None:
            raise RuntimeError("任务无计划事件，无法恢复")
        return plan

    def _plan_dimensions(self) -> list[str]:
        dims: list[str] = []
        for _seq, type_, payload in self.events.replay(self.task_id):
            if type_ == "plan.created":
                dims = [str(d) for d in payload.get("compare_dimensions") or []]
        return dims

    # ------------------------------------------------------------- execute
    def _execute(self, plan: list[dict], cursor: int,
                 stop_after: int | None) -> RunOutcome | None:
        for i in range(cursor, len(plan)):
            if stop_after is not None and i >= stop_after:
                # 测试用：模拟进程崩溃（checkpoint 已在上一子任务后落盘）
                self.events.append(self.task_id, "task.crashed_simulated",
                                   {"at_subtask": i})
                raise _SimulatedCrash
            subtask = plan[i]
            scratch = self.events.scratchpad_for_subtask(self.task_id, i)
            result = self.loop.run_subtask(self.task_id, self.task, subtask, i, scratch)
            if result.exhausted:
                return self._partial_report(plan, i)
            self.events.save_checkpoint(self.task_id, stage="EXECUTE", subtask_cursor=i + 1)

        self.events.save_checkpoint(self.task_id, stage="WRITE", subtask_cursor=len(plan))
        return None

    # ------------------------------------------------------------- write
    def _write(self) -> RunOutcome:
        plan = self._load_plan()
        evidence = self.artifacts.full(self.task_id, type_="evidence")
        ctx = build_context("WRITE", self.task, plan=plan,
                            artifacts=evidence, content_chars=400)
        allowed = sorted(self._known_urls(evidence))
        if allowed:
            ctx += ("\nALLOWED_URLS（参考文献只能逐字使用以下 URL，"
                    "其他任何链接都视为编造）:\n"
                    + "\n".join(f"- {u}" for u in allowed) + "\n")
        dims = self._plan_dimensions()
        dims_line = ("、".join(dims) if dims
                     else "计划未指定时，自行选择最贴合研究主题的 2-3 个维度")
        raw = self._llm_call("WRITE", WRITE_PROMPT.format(dims=dims_line), ctx)
        raw = self._verify_citations(raw, evidence)

        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.report_dir / f"{self.task_id}.md"
        report_path.write_text(raw, encoding="utf-8")
        self.artifacts.write(self.task_id, "report", "最终报告", raw)
        self.events.append(self.task_id, "report.generated", {"path": str(report_path)})
        self.events.save_checkpoint(self.task_id, stage="DONE",
                                    subtask_cursor=len(plan))
        self.events.append(self.task_id, "task.completed",
                           {"subtasks": len(plan), "report": str(report_path)})
        return RunOutcome(status="done", report_path=str(report_path),
                          completed_subtasks=len(plan), total_subtasks=len(plan),
                          gaps=[])

    # ------------------------------------------------- 引用校验（最小 SELF-CHECK）
    def _known_urls(self, evidence: list[dict]) -> set[str]:
        """任务内可验证的 URL 集合：证据工件 refs + 成功抓取过的 URL。"""
        known = {r for a in evidence for r in a.get("refs", [])}
        for _s, t, p in self.events.replay(self.task_id):
            if t == "tool.called" and p.get("tool") == "web_fetch" and p.get("ok"):
                if p.get("args", {}).get("url"):
                    known.add(p["args"]["url"])
        return known

    def _verify_citations(self, report: str, evidence: list[dict]) -> str:
        """报告中的 URL 必须能在任务证据中验证；无法验证的标注出来（不静默删除）。"""
        import re as _re
        known = self._known_urls(evidence)
        urls = set()
        for m in _re.findall(r"https?://[^\s\)\]>\"'。，、（）：]+", report):
            urls.add(m.rstrip(".,;"))
        verified = sorted(u for u in urls if u in known)
        unverified = sorted(u for u in urls if u not in known)
        if unverified:
            report += (
                "\n\n> ⚠️ **引用校验**：以下链接无法在本任务的证据工件/抓取记录中验证，"
                "可能为模型生成，引用前请人工核实：\n"
                + "\n".join(f"> - {u}" for u in unverified)
            )
        self.events.append(self.task_id, "citation.checked", {
            "verified": verified, "unverified": unverified})
        return report

    def _partial_report(self, plan: list[dict], done_up_to: int) -> RunOutcome:
        done = [s["title"] for s in plan[:done_up_to]]
        pending = [s["title"] for s in plan[done_up_to:]]
        evidence = self.artifacts.full(self.task_id, type_="evidence")
        lines = [f"# 研究报告（未完成：预算耗尽，已暂停）", "",
                 f"已完成子任务 {done_up_to}/{len(plan)}：",
                 *[f"- {t}" for t in done], "",
                 "未完成子任务（恢复后继续）：",
                 *[f"- {t}" for t in pending], "", "## 已有证据",
                 *[f"- {a['id']} {a['title']}（refs: {', '.join(a.get('refs', []))}）"
                   for a in evidence]]
        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.report_dir / f"{self.task_id}-partial.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.events.append(self.task_id, "task.paused",
                           {"reason": "budget_exhausted", "done": done_up_to,
                            "report": str(report_path)})
        return RunOutcome(status="paused", report_path=str(report_path),
                          completed_subtasks=done_up_to, total_subtasks=len(plan),
                          gaps=[f"未完成: {t}" for t in pending])
