"""模型网关测试：重试/退避、降级链、熔断、token 预算（全离线，HTTP 可注入）。"""
import json

import httpx
import pytest

from app.runtime.agent_loop import AgentLoop
from app.runtime.budget import Budget
from app.runtime.events import EventStore
from app.runtime.llm import (CircuitBreaker, GatewayError, LLMError,
                            ModelGateway, OpenAICompatibleProvider)
from app.tools.artifact_store import ArtifactStore
from app.tools.registry import ToolRouter
from app.tools.web import MockWebBackend


# ---------------------------------------------------------------- fakes

class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def ok_response(text="hello", tokens_in=10, tokens_out=5):
    return FakeResponse(200, {"choices": [{"message": {"content": text}}],
                              "usage": {"prompt_tokens": tokens_in,
                                        "completion_tokens": tokens_out}})


class FakeHTTP:
    """按脚本依次返回响应或抛异常。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def post(self, url, json=None, headers=None, **kw):
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


def make_provider(script, name="p1", max_retries=2, **kw):
    http = FakeHTTP(script)
    p = OpenAICompatibleProvider(name, "https://fake/v1", "k", "m",
                                 max_retries=max_retries, backoff_base=0.0,
                                 http_client=http, sleep_fn=lambda s: None)
    return p, http


# ---------------------------------------------------------------- tests

def test_retry_on_500_then_success():
    p, http = make_provider([FakeResponse(500), FakeResponse(503), ok_response()])
    text, meta = p.complete("s", "u")
    assert text == "hello"
    assert meta.retries == 2
    assert meta.tokens_in == 10 and meta.tokens_out == 5
    assert http.calls == 3


def test_retry_exhausted_raises():
    p, _ = make_provider([FakeResponse(500)], max_retries=2)
    with pytest.raises(LLMError) as ei:
        p.complete("s", "u")
    assert ei.value.retryable and "重试" in str(ei.value)
    assert "500" in str(ei.value)


def test_timeout_retried_then_success():
    p, _ = make_provider([httpx.TimeoutException("t/o"), ok_response()])
    text, meta = p.complete("s", "u")
    assert text == "hello" and meta.retries == 1


def test_non_retryable_401_immediate_error():
    p, http = make_provider([FakeResponse(401, text="unauthorized")], max_retries=3)
    with pytest.raises(LLMError) as ei:
        p.complete("s", "u")
    assert not ei.value.retryable
    assert http.calls == 1  # 401 不重试


def test_gateway_fallback_chain():
    """主模型持续 500 → 熔断 → 降级到备模型成功。"""
    primary, _ = make_provider([FakeResponse(500)], name="primary", max_retries=0)
    backup, _ = make_provider([ok_response("from-backup", 3, 2)], name="backup")
    gw = ModelGateway([primary, backup], failure_threshold=1, cooldown_seconds=60)

    # 第一次：主失败（1 次即达阈值）→ 降级到备成功
    assert gw.complete("s", "u") == "from-backup"
    assert gw.last_meta.provider == "backup"
    # 主模型熔断打开，后续请求直接跳过主模型
    assert gw.breakers["primary"].is_open
    backup.skip_check = True  # backup 脚本耗尽后仍可用（FakeHTTP 循环最后一项）
    assert gw.complete("s", "u") == "from-backup"   # 这次没再打 primary
    assert gw.breakers["primary"].failures == 1     # 失败数不再增加
    backup2, http2 = make_provider([ok_response("again")], name="backup2")
    gw2 = ModelGateway([primary, backup2], failure_threshold=1)
    primary._http = FakeHTTP([FakeResponse(500)])
    gw2.complete("s", "u")  # primary 失败 → 熔断(阈值1) → backup2
    assert http2.calls == 1  # backup2 只被调一次


def test_gateway_all_failed():
    p1, _ = make_provider([FakeResponse(500)], name="a", max_retries=0)
    p2, _ = make_provider([httpx.ConnectError("down")], name="b", max_retries=0)
    gw = ModelGateway([p1, p2])
    with pytest.raises(GatewayError) as ei:
        gw.complete("s", "u")
    assert "所有 provider 均失败" in str(ei.value)


def test_circuit_breaker_lifecycle():
    now = [0.0]
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=10,
                        clock=lambda: now[0])
    assert cb.allow()
    cb.record_failure()
    assert cb.allow()            # 1 次失败未达阈值
    cb.record_failure()
    assert not cb.allow()        # 熔断
    now[0] = 9.9
    assert not cb.allow()        # 冷却中
    now[0] = 10.0
    assert cb.allow()           # half-open 放行探测
    cb.record_ok()
    assert cb.allow()           # 探测成功 → 关闭


# ---------------------------------------------------------------- token 预算

class MetaLLM:
    """每步消耗 100 token 的假 provider，验证 token 预算收尾。"""
    name = "meta-llm"

    def __init__(self):
        from app.runtime.llm import CallMeta
        self.CallMeta = CallMeta
        self.last_meta = None
        self._n = 0

    def complete(self, system, user):
        self._n += 1
        import json as _json
        raw = _json.dumps({"action": "web_search", "args": {"query": f"q{self._n}"}})
        self.last_meta = self.CallMeta(provider=self.name, model=self.name,
                                       tokens_in=50, tokens_out=50)
        return raw


def test_token_budget_triggers_shutdown(tmp_path):
    events = EventStore(tmp_path / "events.db")
    tools = ToolRouter(MockWebBackend(), ArtifactStore(tmp_path / "artifacts"))
    llm = MetaLLM()
    loop = AgentLoop(llm, tools, events, Budget(max_llm_calls=100,
                                                max_total_tokens=150))
    r = loop.run_subtask("t1", "task", {"title": "s", "focus": "f"}, 0, [])
    # 第 1 次调用后 tokens=100 < 150 可再调；第 2 次后 200 ≥ 150 → 下一步暂停
    assert r.exhausted and "预算" in r.summary
    types = [t for _s, t, _p in events.replay("t1")]
    assert "subtask.paused" in types
    # llm.called 事件带 token 元数据
    llm_events = [p for _s, t, p in events.replay("t1") if t == "llm.called"]
    assert all("tokens_in" in p and "tokens_out" in p for p in llm_events)


def test_plan_limit_trims_plan(tmp_path, monkeypatch):
    """缩小版任务：plan_limit 限制子任务数（mock LLM 遵守 PLAN_LIMIT）。"""
    from app.runtime.model_gateway import MockResearchLLM
    from app.runtime.orchestrator import Orchestrator
    from app.tools.web import MockWebBackend

    storage = tmp_path / "storage"
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    orch = Orchestrator("t-limit", "调研", MockResearchLLM(),
                        ToolRouter(MockWebBackend(), artifacts), events,
                        artifacts, Budget(), storage / "reports",
                        plan_limit=2)
    outcome = orch.run()
    assert outcome.status == "done"
    assert outcome.total_subtasks == 2  # 4 个候选计划截断到 2
    plan_events = [p for _s, t, p in events.replay("t-limit")
                   if t == "plan.created"]
    assert len(plan_events[0]["subtasks"]) == 2
