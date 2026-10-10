"""引用校验（最小 SELF-CHECK）：报告中的 URL 必须能在任务证据中验证。"""
import json

from app.runtime.budget import Budget
from app.runtime.events import EventStore
from app.runtime.orchestrator import Orchestrator
from app.tools.artifact_store import ArtifactStore
from app.tools.registry import ToolRouter
from app.tools.web import MockWebBackend

PROMPT = "调研 Coding Agent 架构"


class ReportLLM:
    """PLAN 正常，WRITE 阶段故意塞一个任务中从未出现过的 URL（模拟模型编造引用）。"""
    name = "report-llm"

    def __init__(self):
        self._n = 0

    def complete(self, system, user):
        self._n += 1
        if "STAGE: PLAN" in user:
            return json.dumps({"subtasks": [
                {"title": "Aider 调研", "focus": "aider coding agent"}]})
        if "STAGE: WRITE" in user:
            return ("# 报告\n\nAider 采用 ReAct 循环（https://aider.chat/docs/）。\n"
                    "另外据称 Devin 支持浏览器（https://totally-fake.example/x）。\n")
        return json.dumps({"action": "finish_subtask", "args": {"summary": "ok"}})


def test_citation_check_flags_fabricated_url(tmp_path):
    storage = tmp_path / "storage"
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    orch = Orchestrator("t-cite", PROMPT, ReportLLM(),
                        ToolRouter(MockWebBackend(), artifacts), events,
                        artifacts, Budget(), storage / "reports")
    outcome = orch.run()
    assert outcome.status == "done"

    report = open(outcome.report_path, encoding="utf-8").read()
    assert "引用校验" in report
    assert "https://totally-fake.example/x" in report  # 被标注而非删除

    checked = [p for _s, t, p in events.replay("t-cite") if t == "citation.checked"]
    assert len(checked) == 1
    # 两个 URL 都没有任务证据支撑（本测试 LLM 未执行任何抓取），均被标注
    assert set(checked[0]["unverified"]) == {
        "https://aider.chat/docs/", "https://totally-fake.example/x"}


def test_citation_check_passes_when_all_refs_verified(tmp_path):
    from app.runtime.model_gateway import MockResearchLLM

    storage = tmp_path / "storage"
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    orch = Orchestrator("t-cite-ok", PROMPT, MockResearchLLM(),
                        ToolRouter(MockWebBackend(), artifacts), events,
                        artifacts, Budget(), storage / "reports", plan_limit=1)
    outcome = orch.run()
    checked = [p for _s, t, p in events.replay("t-cite-ok") if t == "citation.checked"]
    assert len(checked) == 1
    assert checked[0]["unverified"] == []      # mock 报告只引用证据中的 URL
    assert checked[0]["verified"]             # 且有已验证引用
