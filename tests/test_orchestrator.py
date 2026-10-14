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


def _dim_llm(plan_json: str, write_report: str = "# 研究报告\n\n内容"):
    class DimLLM:
        name = "dim"

        def complete(self, system, user):
            if "STAGE: PLAN" in user:
                return plan_json
            if "STAGE: WRITE" in user:
                return write_report
            return json.dumps({"action": "finish_subtask", "args": {"summary": "ok"}})
    return DimLLM()


def test_compare_dimensions_topic_driven(tmp_path):
    """PLAN 产出主题专属对比维度 → 写入事件/工件 → 注入 WRITE 提示词。"""
    plan = json.dumps({
        "subtasks": [{"title": "iPhone 14 调研", "focus": "iphone 14 market share"}],
        "compare_dimensions": ["市场份额", "价格带", "目标人群"]}, ensure_ascii=False)
    storage = tmp_path / "storage"
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    llm = _dim_llm(plan)
    # 捕获 WRITE 阶段提示词
    captured = {}
    orig = llm.complete
    def spy(system, user):
        if "STAGE: WRITE" in user:
            captured["write"] = system
        return orig(system, user)
    llm.complete = spy

    orch = Orchestrator("t-dim", "当前手机市场分析", llm,
                        ToolRouter(MockWebBackend(), artifacts), events,
                        artifacts, Budget(), storage / "reports", plan_limit=1)
    outcome = orch.run()
    assert outcome.status == "done"

    # 事件与工件都带维度
    plan_ev = next(p for _s, t, p in events.replay("t-dim") if t == "plan.created")
    assert plan_ev["compare_dimensions"] == ["市场份额", "价格带", "目标人群"]
    plan_art = artifacts.full("t-dim", type_="plan")[0]["content"]
    assert "市场份额" in plan_art
    # WRITE 提示词用了主题维度，且不再写死 Coding Agent 三维度
    assert "市场份额、价格带、目标人群" in captured["write"]
    assert "任务规划 / 工具使用 / 上下文管理" not in captured["write"]


def test_compare_dimensions_missing_falls_back(tmp_path):
    """旧计划/模型未输出维度 → WRITE 提示词用兑底文案，不报错。"""
    plan = json.dumps({"subtasks": [{"title": "A 调研", "focus": "a"}]}, ensure_ascii=False)
    storage = tmp_path / "storage"
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    llm = _dim_llm(plan)
    captured = {}
    orig = llm.complete
    def spy(system, user):
        if "STAGE: WRITE" in user:
            captured["write"] = system
        return orig(system, user)
    llm.complete = spy
    orch = Orchestrator("t-dim2", "某主题", llm,
                        ToolRouter(MockWebBackend(), artifacts), events,
                        artifacts, Budget(), storage / "reports", plan_limit=1)
    assert orch.run().status == "done"
    assert "自行选择最贴合研究主题" in captured["write"]


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


def test_plan_json_with_trailing_extra_json(tmp_path):
    """回归：贪婪正则 r"\{.*\}" 会从第一个 { 匹配到最后一个 }。

    PLAN 输出“围栏 JSON + 后随第二个 JSON 对象”（GLM 系列常见）时旧实现
    必然 Extra data 失败；新实现按括号平衡只取第一个完整对象。
    """
    from app.runtime.orchestrator import _extract_json
    raw = ('```json\n'
           '{"subtasks": [{"title": "A", "focus": "a"}]}\n'
           '```\n补充维度：{"compare_dimensions": ["价格", "份额"]}')
    obj = _extract_json(raw)
    assert obj["subtasks"] == [{"title": "A", "focus": "a"}]

    # 后随杂文含 } 也不再拉爆范围
    obj2 = _extract_json('计划 {"subtasks": []} 说明：{占位} 结束')
    assert obj2 == {"subtasks": []}

    import pytest
    with pytest.raises(ValueError, match="无有效 JSON"):
        _extract_json("没有 JSON 的输出")


# ---- 时间线复盘 t-eb3dbd0f 的四项修复回归 ----

class _RecordingLLM:
    """委托 MockResearchLLM，但记录每次 complete 的上下文。"""

    name = "rec"

    def __init__(self):
        self._inner = MockResearchLLM()
        self.ctxs = []
        self.last_meta = None

    def complete(self, system, user, timeout=None):
        self.ctxs.append(user)
        text = self._inner.complete(system, user)
        self.last_meta = self._inner.last_meta
        return text


def _make_orch_with(tmp_path, task_id, llm, **kw):
    storage = tmp_path / "storage"
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    orch = Orchestrator(task_id, PROMPT, llm,
                        ToolRouter(MockWebBackend(), artifacts), events,
                        artifacts, Budget(), storage / "reports", **kw)
    return orch, artifacts


def test_agent_max_steps_wiring(tmp_path):
    """agent_max_steps 透传给 AgentLoop（生产链路由 AppConfig 注入默认 20）。"""
    orch, artifacts = _make_orch_with(tmp_path, "t-ms", MockResearchLLM())
    assert orch.loop.max_steps == 12  # orchestrator 层默认值
    orch2, _ = _make_orch_with(tmp_path, "t-ms2", MockResearchLLM(),
                               agent_max_steps=20)
    assert orch2.loop.max_steps == 20


def test_write_evidence_chars_controls_truncation(tmp_path):
    """WRITE 阶段证据注入字符数：400 截断、4000 全量（旧硬编码 400 卡报告质量）。"""
    long_text = "X" * 1000

    def run(sub, chars):
        storage = tmp_path / sub
        events = EventStore(storage / "events.db")
        artifacts = ArtifactStore(storage / "artifacts")
        artifacts.write("t-ev", "evidence", "long", long_text, ["https://x"])
        llm = _RecordingLLM()
        orch = Orchestrator("t-ev", PROMPT, llm,
                            ToolRouter(MockWebBackend(), artifacts), events,
                            artifacts, Budget(), storage / "reports",
                            write_evidence_chars=chars)
        assert orch.run().status == "done"
        write_ctx = [c for c in llm.ctxs if "STAGE: WRITE" in c]
        assert len(write_ctx) == 1
        return write_ctx[0]

    assert "...[截断]" in run("a", 400)
    assert long_text in run("b", 4000)  # 4000 字符窗口装得下 1000 字全文


class _TwoArgLLM:
    """旧式两参 LLM（自定义 fake / 用户代码）：不接受 timeout kwarg。"""

    name = "two-arg"

    def __init__(self):
        self._inner = MockResearchLLM()
        self.last_meta = None

    def complete(self, system, user):
        text = self._inner.complete(system, user)
        self.last_meta = self._inner.last_meta
        return text


def test_write_timeout_falls_back_for_two_arg_llm(tmp_path):
    """write_timeout 已配置但 LLM 不支持该参数 → 回退默认窗口而非崩溃。"""
    orch, _ = _make_orch_with(tmp_path, "t-2a", _TwoArgLLM(), write_timeout=270)
    assert orch.run().status == "done"


class _TimeoutProbe:
    """真实 provider 形状（返回 (text, meta)）：记录收到的 timeout。"""

    name = "probe"

    def __init__(self):
        self._inner = MockResearchLLM()
        self.calls = []
        self.last_meta = None

    def complete(self, system, user, timeout=None):
        self.calls.append((user.splitlines()[0] if user else "", timeout))
        text, meta = self._inner.complete(system, user), self._inner.last_meta
        self.last_meta = meta
        return text, meta


def test_write_timeout_passed_to_provider(tmp_path):
    """WRITE 调用的读超时窗口透传到 provider；PLAN/EXECUTE 不受影响。"""
    from app.runtime.llm import ModelGateway
    probe = _TimeoutProbe()
    gw = ModelGateway([probe])
    storage = tmp_path / "storage"
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    orch = Orchestrator("t-to", PROMPT, gw,
                        ToolRouter(MockWebBackend(), artifacts), events,
                        artifacts, Budget(), storage / "reports",
                        write_timeout=270)
    assert orch.run().status == "done"
    write_calls = [(s, t) for s, t in probe.calls if t is not None]
    assert [t for _, t in write_calls] == [270]  # 仅 WRITE 一次
    non_write = [t for s, t in probe.calls if t is None]
    assert len(non_write) >= 2  # PLAN + EXECUTE 若干次，均用 provider 默认


def test_write_zero_evidence_disclaimer(tmp_path):
    """零证据工件 → WRITE 上下文明示「缺乏一手证据」边界（t-a58c3f76 回归：
    全程未调 write_artifact，报告 2.6k 字符塌方且引用无任务内依据）。"""
    import json as _json
    seen: dict = {}

    class _FinishNowLLM:
        name = "finish-now"
        last_meta = None

        def complete(self, system, user, timeout=None):
            if "STAGE: PLAN" in user:
                return _json.dumps({"subtasks": [{"title": "A", "focus": "a"}]})
            if "STAGE: WRITE" in user:
                seen["write_ctx"] = user
                return "本报告缺乏一手证据\n\n# 报告\n内容\n"
            return _json.dumps({"action": "finish_subtask",
                                "args": {"summary": "done"}})

    storage = tmp_path / "storage"
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    orch = Orchestrator("t-ze", PROMPT, _FinishNowLLM(),
                        ToolRouter(MockWebBackend(), artifacts), events,
                        artifacts, Budget(), storage / "reports")
    assert orch.run().status == "done"
    assert "没有写入任何证据工件" in seen["write_ctx"]
    assert "严禁凭记忆编造链接" in seen["write_ctx"]
