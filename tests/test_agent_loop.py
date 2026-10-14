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


def test_system_prompt_documents_tool_args():
    """回归：SYSTEM_PROMPT 必须给出工具参数表。

    时间线复盘 t-eb3dbd0f：参数表缺失 → 模型盲猜 read_artifact 的参数名
    （三次空 id 调用，浪费 5/36 步）。参数表 + "不要原样重试" 规则防复发。
    """
    from app.runtime.agent_loop import SYSTEM_PROMPT
    assert "参数表" in SYSTEM_PROMPT
    for key in ('"query"', '"url"', '"content"', '"refs"', '"id"', '"summary"'):
        assert key in SYSTEM_PROMPT, f"参数表缺少 {key}"
    assert "不要原样重试" in SYSTEM_PROMPT


# ---- 时间线复盘 t-a58c3f76 回归：顶层丢参 + 退化循环 + 诊断可见性 ----

def test_top_level_args_merged(tmp_path):
    """模型把 query 写在顶层（不进 args）→ 解析器合并进 args，工具正常执行。

    t-a58c3f76：41/47 次 web_search 因此静默丢参失败，模型反复重试同一结构。
    """
    events, _a, _llm, loop = make_env(tmp_path, [
        '{"action":"web_search","query":"随机森林 优缺点"}',
        act("finish_subtask", summary="done"),
    ])
    r = loop.run_subtask("t1", "task", {"title": "s", "focus": "f"}, 0, [])
    assert not r.exhausted and r.summary == "done"
    tool_ev = [p for _s, t, p in events.replay("t1") if t == "tool.called"][0]
    assert tool_ev["ok"] and tool_ev["args"]["query"] == "随机森林 优缺点"


def test_repeat_fail_guard_breaks_degenerate_loop(tmp_path):
    """同一失败调用重复 → 第 2 次失败即注入 REPEAT_FAIL 换路提示（进 scratch/上下文）。"""
    _e, _a, llm, loop = make_env(tmp_path, [
        act("web_search"), act("web_search"), act("web_search"),
        act("finish_subtask", summary="ok"),
    ])
    scratch = []
    loop.run_subtask("t1", "task", {"title": "s", "focus": "f"}, 0, scratch)
    assert any("REPEAT_FAIL" in s for s in scratch)
    assert "REPEAT_FAIL" in llm.calls[2]  # 第 3 轮上下文模型实际可见


def test_max_steps_without_evidence_flagged(tmp_path):
    """max_steps 收尾且全程零 write_artifact → summary/gaps 明示「未写入证据」。"""
    _e, _a, _llm, loop = make_env(tmp_path, [act("web_search", query="x")] * 13)
    r = loop.run_subtask("t1", "task", {"title": "s", "focus": "f"}, 0, [])
    assert r.steps == 12 and r.steps == loop.max_steps  # 确实撞顶
    assert "未写入任何证据工件" in r.summary and "未写入证据" in r.gaps


def test_llm_called_event_records_raw_head(tmp_path):
    """llm.called 事件带原文前 300 字符（静默丢参问题在事件流里可见）。"""
    events, _a, _llm, loop = make_env(tmp_path, [act("finish_subtask", summary="done")])
    loop.run_subtask("t1", "task", {"title": "s", "focus": "f"}, 0, [])
    llm_ev = [p for _s, t, p in events.replay("t1") if t == "llm.called"][0]
    assert "finish_subtask" in llm_ev["raw_head"]
