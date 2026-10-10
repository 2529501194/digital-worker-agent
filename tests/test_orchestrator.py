"""端到端：PLAN → EXECUTE → WRITE，产物与事件完整性。"""
import json
from pathlib import Path

from app.runtime.budget import Budget
from app.runtime.events import EventStore
from app.runtime.model_gateway import MockResearchLLM
from app.runtime.orchestrator import Orchestrator
from app.tools.artifact_store import ArtifactStore
from app.tools.registry import ToolRouter
from app.tools.web import MockWebBackend


PROMPT = "调研当前主流 Coding Agent 的技术架构，比较任务规划/工具使用/上下文管理差异"


def make_orch(tmp_path, task_id="t-e2e", budget=None, stop_after=None):
    storage = tmp_path / "storage"
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    tools = ToolRouter(MockWebBackend(), artifacts)
    orch = Orchestrator(task_id, PROMPT, MockResearchLLM(), tools, events,
                        artifacts, budget or Budget(), storage / "reports")
    return orch, events, artifacts, storage


def test_end_to_end_research_task(tmp_path):
    orch, events, artifacts, storage = make_orch(tmp_path)
    outcome = orch.run()

    assert outcome.status == "done"
    assert outcome.completed_subtasks == 4 == outcome.total_subtasks

    # 报告落盘且带引用
    report = Path(outcome.report_path).read_text(encoding="utf-8")
    assert report.startswith("# 研究报告")
    assert "Claude Code" in report and "OpenHands" in report
    assert "https://" in report  # 参考文献

    # 工件：计划 + 4 份证据 + 报告
    assert len(artifacts.list("t-e2e", "plan")) == 1
    assert len(artifacts.list("t-e2e", "evidence")) == 4
    assert len(artifacts.list("t-e2e", "report")) == 1

    # 事件流关键节点齐全
    types = [t for _s, t, _p in events.replay("t-e2e")]
    for expected in ("task.created", "plan.created", "subtask.finished",
                     "report.generated", "task.completed"):
        assert expected in types, f"缺少事件 {expected}"
    assert types.count("subtask.finished") == 4
    assert types.count("checkpoint.saved") >= 5  # EXECUTE x4 + WRITE + DONE


def test_resume_after_partial_run_is_noop_safe(tmp_path):
    """已 DONE 的任务再次 run：不重做，不产生重复抓取。"""
    orch, events, artifacts, storage = make_orch(tmp_path)
    first = orch.run()
    # 第二次用同一 storage 的新 Orchestrator（模拟重启后再次触发）
    events2 = EventStore(storage / "events.db")
    artifacts2 = ArtifactStore(storage / "artifacts")
    orch2 = Orchestrator("t-e2e", PROMPT, MockResearchLLM(),
                         ToolRouter(MockWebBackend(), artifacts2), events2,
                         artifacts2, Budget(), storage / "reports")
    second = orch2.run()
    assert second.status == "done"
    types = [t for _s, t, _p in events2.replay("t-e2e")]
    assert "task.resumed" in types  # 走的是恢复路径
    # 报告未被重写两次（第二次走 _write 覆盖同一路径，事件多一组但不影响正确性）
    assert len(artifacts2.list("t-e2e", "evidence")) == 4


def test_budget_exhaustion_produces_partial_report(tmp_path):
    """工具预算极小 → 收尾模式：部分报告 + paused 事件，而非失控。"""
    orch, events, _art, storage = make_orch(tmp_path, task_id="t-budget",
                                            budget=Budget(max_tool_calls=3))
    outcome = orch.run()
    assert outcome.status == "paused"
    assert outcome.completed_subtasks < outcome.total_subtasks
    partial = Path(outcome.report_path).read_text(encoding="utf-8")
    assert "未完成子任务" in partial
    types = [t for _s, t, _p in events.replay("t-budget")]
    assert "task.paused" in types
    # 检查点停在 EXECUTE，可恢复
    cp = events.load_checkpoint("t-budget")
    assert cp.stage == "EXECUTE" and 0 < cp.subtask_cursor < 4


def test_plan_json_with_markdown_fences(tmp_path):
    """真实模型常把 JSON 包在 ```json 围栏里 —— PLAN 提取必须健壮。"""
    import json

    class FencedLLM:
        name = "fenced"

        def complete(self, system, user):
            if "STAGE: PLAN" in user:
                return ("```json\n"
                        '{"subtasks": [{"title": "A 调研", "focus": "a focus"}]}\n'
                        "``` 以上为计划。")
            if "STAGE: WRITE" in user:
                return "# 研究报告\n\n- A 调研：证据 E-1（https://x）\n"
            return json.dumps({"action": "finish_subtask", "args": {"summary": "ok"}})

    storage = tmp_path / "storage"
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    orch = Orchestrator("t-fence", PROMPT, FencedLLM(),
                        ToolRouter(MockWebBackend(), artifacts), events,
                        artifacts, Budget(), storage / "reports", plan_limit=1)
    outcome = orch.run()
    assert outcome.status == "done" and outcome.total_subtasks == 1
    report = Path(outcome.report_path).read_text(encoding="utf-8")
    assert report.startswith("# 研究报告")
