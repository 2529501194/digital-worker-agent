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
