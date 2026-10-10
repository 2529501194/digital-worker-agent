"""Agent 循环单测：动作解析失败恢复、工具调用写事件、scratchpad 回写。"""
import json

import pytest

from app.runtime.agent_loop import AgentLoop
from app.runtime.budget import Budget
from app.runtime.events import EventStore
from app.runtime.model_gateway import ScriptedLLM
from app.tools.artifact_store import ArtifactStore
from app.tools.registry import ToolRouter
from app.tools.web import MockWebBackend

URL = "https://docs.claude-code.dev/arch"


def make_env(tmp_path, responses):
    events = EventStore(tmp_path / "events.db")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    tools = ToolRouter(MockWebBackend(), artifacts)
    llm = ScriptedLLM(responses)
    loop = AgentLoop(llm, tools, events, Budget())
    return events, artifacts, llm, loop


def act(name, **args):
    return json.dumps({"action": name, "args": args}, ensure_ascii=False)


def test_happy_path_search_fetch_evidence_finish(tmp_path):
    events, artifacts, _llm, loop = make_env(tmp_path, [
        act("web_search", query="claude code"),
        act("web_fetch", url=URL),
        act("write_artifact", type="evidence", title="cc", content="evidence", refs=[URL]),
        act("finish_subtask", summary="done"),
    ])
    r = loop.run_subtask("t1", "task", {"title": "s", "focus": "f"}, 0, [])
    assert not r.exhausted and r.summary == "done"
    # 工具事件 + 工件落盘
    types = [t for _s, t, _p in events.replay("t1")]
    assert types.count("tool.called") == 3
    assert artifacts.list("t1", "evidence")[0]["id"] == "E-1"


def test_parse_error_recovered(tmp_path):
    """LLM 先吐坏 JSON → 解析失败写事件并注入纠错 → 下一轮成功。"""
    events, _art, llm, loop = make_env(tmp_path, [
        "完全不是 JSON 的废话",
        act("web_search", query="cursor"),
        act("finish_subtask", summary="ok"),
    ])
    r = loop.run_subtask("t1", "task", {"title": "s", "focus": "f"}, 0, [])
    assert r.summary == "ok"
    assert [t for _s, t, _p in events.replay("t1")].count("llm.parse_error") == 1
    # 第二次 LLM 调用的上下文里带纠错提示
    assert "PARSE_ERROR" in llm.calls[1]


def test_persistent_parse_error_aborts(tmp_path):
    """持续解析失败 → 中止子任务而不是死循环。"""
    events, _art, _llm, loop = make_env(
        tmp_path, ["bad1", "bad2", "bad3", "bad4"])
    r = loop.run_subtask("t1", "task", {"title": "s", "focus": "f"}, 0, [])
    assert r.exhausted and "无法解析" in r.summary
    types = [t for _s, t, _p in events.replay("t1")]
    assert "subtask.aborted" in types


def test_tool_error_visible_in_scratchpad(tmp_path):
    """抓取 404 → 错误进入 scratchpad，下一轮 LLM 可见（可换路）。"""
    events, _art, llm, loop = make_env(tmp_path, [
        act("web_fetch", url="https://no-such.example/x"),
        act("finish_subtask", summary="gave up", gaps=["抓取失败"]),
    ])
    r = loop.run_subtask("t1", "task", {"title": "s", "focus": "f"}, 0, [])
    assert r.gaps == ["抓取失败"]
    assert "ERROR" in llm.calls[1]  # 第二次调用上下文含错误观察


def test_max_steps_guard(tmp_path):
    """LLM 一直不 finish → 步数上限收尾，不无限循环。"""
    loop_script = [act("web_search", query=f"q{i}") for i in range(20)]
    events, _art, _llm, loop = make_env(tmp_path, loop_script)
    loop.max_steps = 6
    r = loop.run_subtask("t1", "task", {"title": "s", "focus": "f"}, 0, [])
    assert r.steps == 6 and "步数上限" in r.summary


def test_budget_exhaustion(tmp_path):
    events, _art, _llm, loop = make_env(tmp_path, [act("web_search", query="x")] * 50)
    loop.budget = Budget(max_llm_calls=2, max_tool_calls=100)
    r = loop.run_subtask("t1", "task", {"title": "s", "focus": "f"}, 0, [])
    assert r.exhausted and "预算" in r.summary
    types = [t for _s, t, _p in events.replay("t1")]
    assert "subtask.paused" in types
