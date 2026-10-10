"""崩溃恢复：kill 模拟（stop_after）→ 新进程（新 Orchestrator，同 storage）断点续跑。

验证：
1. 从 checkpoint 恢复，跳过已完成子任务；
2. 已完成子任务的外部调用不重做（web_fetch 事件总数 == 子任务数，不翻倍）;
3. 最终报告完整产出。
"""
from pathlib import Path

from app.runtime.budget import Budget
from app.runtime.events import EventStore
from app.runtime.model_gateway import MockResearchLLM
from app.runtime.orchestrator import Orchestrator
from app.tools.artifact_store import ArtifactStore
from app.tools.registry import ToolRouter
from app.tools.web import MockWebBackend

PROMPT = "调研主流 Coding Agent 架构"
TASK = "t-crash"


def make_orch(storage, task_id=TASK):
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    # 每次"进程启动"都是新的内存态（工具缓存、预算）—— 与真实重启一致
    return Orchestrator(task_id, PROMPT, MockResearchLLM(),
                        ToolRouter(MockWebBackend(), artifacts), events,
                        artifacts, Budget(), storage / "reports")


def test_crash_then_resume(tmp_path):
    storage = tmp_path / "storage"

    # 进程 1：跑到第 2 个子任务后"崩溃"
    orch1 = make_orch(storage)
    outcome1 = orch1.run(stop_after_subtasks=2)
    # 崩溃模拟下 run 返回 None（进程死掉，没有正常返回）
    assert outcome1 is None

    events1 = EventStore(storage / "events.db")
    cp = events1.load_checkpoint(TASK)
    assert cp is not None and cp.stage == "EXECUTE" and cp.subtask_cursor == 2
    fetch_events_1 = [t for _s, t, _p in events1.replay(TASK)
                      if t == "tool.called"]
    assert len(fetch_events_1) == 6  # 2 子任务 x 3 工具调用

    # 进程 2：新 Orchestrator（全新内存态），同一 storage → 断点续跑
    orch2 = make_orch(storage)
    outcome2 = orch2.run()

    assert outcome2.status == "done"
    assert outcome2.completed_subtasks == 4

    events2 = EventStore(storage / "events.db")
    types = [t for _s, t, _p in events2.replay(TASK)]
    assert "task.resumed" in types

    # 关键：恢复后前 2 个子任务没有重做 —— web_fetch 事件总数仍为 4（每子任务 1 次）
    fetch_events_all = [p for _s, t, p in events2.replay(TASK) if t == "tool.called"
                        and p.get("tool") == "web_fetch"]
    assert len(fetch_events_all) == 4, f"外部调用被重复执行: {len(fetch_events_all)}"

    # 报告完整
    report = Path(outcome2.report_path).read_text(encoding="utf-8")
    assert "OpenHands" in report and "Claude Code" in report
    assert "参考文献" in report


def test_scratchpad_reconstructed_from_events(tmp_path):
    """恢复后，未完成子任务的 scratchpad 能从事件流重建（循环内记忆不丢）。"""
    storage = tmp_path / "storage"
    orch1 = make_orch(storage)
    orch1.run(stop_after_subtasks=1)

    events = EventStore(storage / "events.db")
    scratch = events.scratchpad_for_subtask(TASK, 0)
    assert any(l.startswith("web_search") for l in scratch)
    assert any(l.startswith("web_fetch") and "OK" in l for l in scratch)
    assert any(l.startswith("write_artifact") for l in scratch)
