"""Web 服务端到端（全 mock，零外部网络）：建任务 → 轮询事件 → 报告 → 列表 → 恢复守卫。"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from app.runtime.model_gateway import MockResearchLLM
from app.tools.web import MockWebBackend


def _req(base: str, method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _start(tmp_path, llm=None, web=None):
    from app.web.server import create_server
    server, mgr = create_server(tmp_path / "storage", port=0,
                                llm=llm or MockResearchLLM(), web=web or MockWebBackend())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, mgr, f"http://127.0.0.1:{server.server_address[1]}"


def _wait_done(base: str, tid: str, timeout: float = 60.0) -> str:
    since, status = 0, "running"
    deadline = time.time() + timeout
    while time.time() < deadline:
        code, body = _req(base, "GET", f"/api/tasks/{tid}/timeline?since={since}")
        assert code == 200
        data = json.loads(body)
        since += len(data["events"])
        status = data["status"]
        if status in ("done", "failed", "paused"):
            break
        time.sleep(0.2)
    return status


def test_serve_end_to_end(tmp_path):
    server, mgr, base = _start(tmp_path)
    try:
        # 首页
        code, html = _req(base, "GET", "/")
        assert code == 200 and "研究数字员工" in html

        # 状态
        code, body = _req(base, "GET", "/api/status")
        assert code == 200 and json.loads(body)["web"] == "mock"

        # 创建任务
        code, body = _req(base, "POST", "/api/tasks",
                          {"prompt": "调研 X", "task_id": "t-web1", "limit_subtasks": 2})
        assert code == 202 and json.loads(body)["task_id"] == "t-web1"

        # 等待完成
        assert _wait_done(base, "t-web1") == "done"

        # 报告
        code, report = _req(base, "GET", "/api/tasks/t-web1/report")
        assert code == 200 and report.startswith("#")

        # 事件流含终态事件
        code, body = _req(base, "GET", "/api/tasks/t-web1/timeline")
        types = [e[1] for e in json.loads(body)["events"]]
        assert "task.completed" in types and "plan.created" in types

        # 列表
        code, body = _req(base, "GET", "/api/tasks")
        tasks = {t["task_id"]: t for t in json.loads(body)["tasks"]}
        assert tasks["t-web1"]["status"] == "done" and tasks["t-web1"]["has_report"]

        # 已完成任务不可恢复
        code, body = _req(base, "POST", "/api/tasks/t-web1/resume")
        assert code == 400
        # 不存在任务恢复 → 400（无检查点）
        code, _ = _req(base, "POST", "/api/tasks/no-such/resume")
        assert code == 400
    finally:
        server.shutdown()
        server.server_close()


class _SlowMock:
    """包裹 Mock LLM 加延时，制造"运行中"窗口用于 409 测试。"""

    name = "slow-mock"

    def __init__(self):
        self._inner = MockResearchLLM()
        self.last_meta = None

    def complete(self, system, user):
        time.sleep(0.4)
        text = self._inner.complete(system, user)
        self.last_meta = self._inner.last_meta
        return text


class _LoopingLLM:
    """EXECUTE 阶段永远只调 web_search → 每个子任务都以 max_steps 收尾（无 subtask.finished）。"""

    name = "looping"

    def __init__(self):
        self.last_meta = None

    def complete(self, system, user):
        if "STAGE: PLAN" in user:
            import json as _json
            return _json.dumps({"subtasks": [{"title": "A 调研", "focus": "a"}]},
                               ensure_ascii=False)
        if "STAGE: WRITE" in user:
            return "# 研究报告\n\n内容"
        return '{"action": "web_search", "args": {"query": "a"}}'


def test_max_steps_subtask_counts_as_done(tmp_path):
    """子任务以 subtask.max_steps 收尾（无 subtask.finished）时，进度也要计入已完成。"""
    server, mgr, base = _start(tmp_path, llm=_LoopingLLM())
    try:
        code, body = _req(base, "POST", "/api/tasks",
                          {"prompt": "p", "task_id": "t-loop", "limit_subtasks": 1})
        assert code == 202
        assert _wait_done(base, "t-loop") == "done"
        code, body = _req(base, "GET", "/api/tasks")
        t = next(x for x in json.loads(body)["tasks"] if x["task_id"] == "t-loop")
        assert t["subtasks_done"] == 1 and t["subtasks_total"] == 1, t
    finally:
        server.shutdown()
        server.server_close()


class _FlakyPlanLLM:
    """PLAN 阶段前 N 次 LLM 调用抛异常（模拟崩溃/故障），之后恢复。

    用于模拟“任务在 PLAN 阶段跑崩”：无检查点落盘（检查点在 _plan 末尾才写），
    任务以 task.failed 收尾；恢复时整体重跑，重新生成计划即可跑完。
    """

    name = "flaky-plan"

    def __init__(self, fail_first: int = 1):
        self._inner = MockResearchLLM()
        self._last_meta = None
        self._plan_calls = 0
        self.fail_first = fail_first
        self.last_meta = None

    def complete(self, system, user):
        import json as _json
        if "STAGE: PLAN" in user:
            self._plan_calls += 1
            if self._plan_calls <= self.fail_first:
                raise RuntimeError("simulated crash during PLAN")
            return _json.dumps({"subtasks": [{"title": "A 调研", "focus": "a"}]},
                               ensure_ascii=False)
        text = self._inner.complete(system, user)
        self._last_meta = self._inner.last_meta
        self.last_meta = self._last_meta
        return text


def test_resume_without_checkpoint_reruns_from_scratch(tmp_path):
    """PLAN 阶段崩溃（无检查点）→ 前端恢复入口可用：整体重跑直至完成。"""
    server, mgr, base = _start(tmp_path, llm=_FlakyPlanLLM(fail_first=1))
    try:
        # 第一次运行：PLAN 阶段崩溃 → task.failed，无检查点
        code, body = _req(base, "POST", "/api/tasks",
                          {"prompt": "调研 X", "task_id": "t-crash", "limit_subtasks": 1})
        assert code == 202
        status = _wait_done(base, "t-crash")
        assert status == "failed"

        code, body = _req(base, "GET", "/api/tasks")
        t = next(x for x in json.loads(body)["tasks"] if x["task_id"] == "t-crash")
        assert t["status"] == "failed" and t["resumable"] is True
        assert t["has_checkpoint"] is False, t

        # 恢复：无检查点 → 取回 prompt 整体重跑（此时 LLM 已恢复）
        code, body = _req(base, "POST", "/api/tasks/t-crash/resume")
        assert code == 202 and json.loads(body)["status"] == "running"
        assert _wait_done(base, "t-crash") == "done"

        # 重跑后事件流：两次 task.created 之后的计划/报告齐全
        code, body = _req(base, "GET", "/api/tasks/t-crash/timeline")
        types = [e[1] for e in json.loads(body)["events"]]
        assert types.count("task.failed") == 1
        assert "plan.created" in types and "task.completed" in types

        code, report = _req(base, "GET", "/api/tasks/t-crash/report")
        assert code == 200 and report.startswith("#")

        # 恢复入口：已完成 → 400；不存在任务 → 400（无 task.created 可取回）
        code, _ = _req(base, "POST", "/api/tasks/t-crash/resume")
        assert code == 400
        code, _ = _req(base, "POST", "/api/tasks/ghost/resume")
        assert code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_task_running_conflict(tmp_path):
    server, mgr, base = _start(tmp_path, llm=_SlowMock())
    try:
        code, body = _req(base, "POST", "/api/tasks", {"prompt": "慢任务", "task_id": "t-busy"})
        assert code == 202
        # 任务仍在运行 → 409
        code, body = _req(base, "POST", "/api/tasks", {"prompt": "再来", "task_id": "t-busy"})
        assert code == 409
        code, body = _req(base, "POST", "/api/tasks/t-busy/resume")
        assert code == 409
        # 空 prompt → 400
        code, _ = _req(base, "POST", "/api/tasks", {"prompt": "  "})
        assert code == 400
        # 等它跑完，之后可恢复入口正常（done → 400）
        assert _wait_done(base, "t-busy", timeout=60) == "done"
        code, _ = _req(base, "POST", "/api/tasks/t-busy/resume")
        assert code == 400
    finally:
        server.shutdown()
        server.server_close()


# ---- 模型配置 API（/api/models*） ----

_ENV_KEYS = ("RE_EMP_LLM_API_KEY", "OPENAI_API_KEY", "RE_EMP_LLM_FALLBACK_API_KEY",
             "LANGSEARCH_API_KEY", "BRAVE_API_KEY", "SERPAPI_API_KEY",
             "RE_EMP_YOU_MCP", "RE_EMP_LLM_NAME", "RE_EMP_LLM_MODEL")


def _start_open(tmp_path, monkeypatch):
    """注册表模式：llm/web 均不注入（先清掉环境变量保证确定性）。"""
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    from app.web.server import create_server
    server, mgr = create_server(tmp_path / "storage", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, mgr, f"http://127.0.0.1:{server.server_address[1]}"


def test_models_api_full_cycle(tmp_path, monkeypatch):
    server, mgr, base = _start_open(tmp_path, monkeypatch)
    try:
        # 初始视图：空链（mock）+ you-mcp（默认启用）
        code, body = _req(base, "GET", "/api/models")
        assert code == 200
        d = json.loads(body)
        assert d["llm"]["chain"] == [] and d["llm"]["locked"] is False
        assert d["web"]["active"] == "you-mcp"

        # 添加两个模型（201）
        code, body = _req(base, "POST", "/api/models/llm",
                          {"name": "primary", "model": "glm-5.3", "key": "sk-1",
                           "base_url": "https://zhipu/v1"})
        assert code == 201 and json.loads(body)["added"]["name"] == "primary"
        code, _ = _req(base, "POST", "/api/models/llm",
                       {"name": "fallback", "model": "gpt-4o-mini", "key": "sk-2"})
        assert code == 201
        assert _req(base, "POST", "/api/models/llm",
                    {"name": "primary", "model": "x"})[0] == 400  # 重名
        assert _req(base, "POST", "/api/models/llm",
                    {"name": "n", "model": "  "})[0] == 400       # 空模型名

        # 重排：fallback 升为主模型；/api/status 跟随
        code, body = _req(base, "POST", "/api/models/llm/order",
                          {"order": ["fallback", "primary"]})
        assert code == 200
        assert [c["name"] for c in json.loads(body)["chain"]] == \
            ["fallback", "primary"]
        assert json.loads(_req(base, "GET", "/api/status")[1])["llm"] == \
            "fallback → primary"
        assert _req(base, "POST", "/api/models/llm/order",
                    {"order": ["fallback"]})[0] == 400  # 不是完整重排

        # 修改 key；未知模型 400
        code, _ = _req(base, "PUT", "/api/models/llm/primary", {"key": "sk-1-new"})
        assert code == 200
        assert _req(base, "PUT", "/api/models/llm/ghost", {"model": "m"})[0] == 400

        # 删除；未知 400
        code, body = _req(base, "DELETE", "/api/models/llm/fallback")
        assert code == 200 and json.loads(body) == {"removed": "fallback"}
        assert _req(base, "DELETE", "/api/models/llm/ghost")[0] == 400

        # Web 搜索 key：设置 brave → 真实后端；未知后端 400
        code, body = _req(base, "POST", "/api/models/web",
                          {"backend": "brave", "key": "bk"})
        assert code == 200 and json.loads(body)["active"] == "brave+http"
        assert _req(base, "POST", "/api/models/web",
                    {"backend": "google", "key": "x"})[0] == 400

        # 持久化：同一 storage 新 manager 读到全部变更（重启不丢）
        from app.web.server import TaskManager
        mgr2 = TaskManager(tmp_path / "storage")
        assert [s.name for s in mgr2.model_registry.specs] == ["primary"]
        assert mgr2.model_registry.specs[0].api_key == "sk-1-new"
        assert mgr2.web_registry.keys["brave"] == "bk"
    finally:
        server.shutdown()
        server.server_close()


def test_models_api_locked_when_injected(tmp_path):
    """llm/web 注入（测试/演示模式）→ 模型配置只读，变更一律 409。"""
    server, mgr, base = _start(tmp_path)
    try:
        assert mgr.model_registry is None and mgr.web_registry is None
        d = json.loads(_req(base, "GET", "/api/models")[1])
        assert d["llm"]["locked"] is True and d["web"]["locked"] is True
        assert d["llm"]["chain"] == []  # MockResearchLLM 无 providers
        assert d["web"]["active"] == "mock"
        assert _req(base, "POST", "/api/models/llm",
                    {"name": "x", "model": "m"})[0] == 409
        assert _req(base, "POST", "/api/models/llm/order",
                    {"order": ["x"]})[0] == 409
        assert _req(base, "PUT", "/api/models/llm/x", {"model": "m"})[0] == 409
        assert _req(base, "DELETE", "/api/models/llm/x")[0] == 409
        assert _req(base, "POST", "/api/models/web",
                    {"backend": "brave", "key": "k"})[0] == 409
    finally:
        server.shutdown()
        server.server_close()
