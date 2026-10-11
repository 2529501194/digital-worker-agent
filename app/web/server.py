"""Web 服务：stdlib ThreadingHTTPServer + JSON API + 单页前端（零新增依赖）。

    python -m app.cli serve --port 8000

API：
    GET  /                    前端单页
    GET  /api/status          当前后端（LLM 链 / 搜索后端）
    GET  /api/tasks           任务列表 + 状态 + 子任务进度
    POST /api/tasks           {prompt, task_id?, limit_subtasks?} 后台线程运行
    POST /api/tasks/{id}/resume   从检查点恢复
    GET  /api/tasks/{id}/timeline?since=0  事件流（增量拉取）
    GET  /api/tasks/{id}/report        报告 Markdown（未生成 404）
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from ..config import AppConfig, build_llm, build_web
from ..runtime.budget import Budget
from ..runtime.events import EventStore
from ..runtime.orchestrator import Orchestrator
from ..tools.artifact_store import ArtifactStore
from ..tools.registry import RESEARCHER_WHITELIST, ToolRouter


class TaskRunningError(Exception):
    pass


class NoCheckpointError(Exception):
    pass


class TaskManager:
    """共享资源 + 后台任务线程管理。一个 task_id 同时只允许一个运行线程。"""

    def __init__(self, storage_root: str | Path, llm=None, web=None,
                 verbose: bool = False):
        self.storage = Path(storage_root)
        self.events = EventStore(self.storage / "events.db")
        self.artifacts = ArtifactStore(self.storage / "artifacts")
        if llm is None or web is None:
            cfg = AppConfig.from_env()
            llm = llm or build_llm(cfg, verbose=verbose)
            web = web or build_web(cfg, self.storage, verbose=verbose)
        self.llm, self.web = llm, web
        self.tools = ToolRouter(web, self.artifacts, RESEARCHER_WHITELIST)
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self.index_html = (Path(__file__).with_name("index.html")
                          .read_text(encoding="utf-8"))
        if hasattr(llm, "providers"):  # ModelGateway：展示降级链
            self.llm_desc = " → ".join(p.name for p in llm.providers)
        else:
            self.llm_desc = getattr(llm, "name", "mock")

    # ---- 任务 ----
    def start_task(self, task_id: str, prompt: str, limit_subtasks: int = 3) -> None:
        with self._lock:
            th = self._threads.get(task_id)
            if th is not None and th.is_alive():
                raise TaskRunningError(task_id)

            def run() -> None:
                orch = Orchestrator(
                    task_id=task_id, task_prompt=prompt, llm=self.llm,
                    tools=self.tools, events=self.events,
                    artifacts=self.artifacts, budget=Budget(),
                    report_dir=self.storage / "reports", plan_limit=limit_subtasks)
                try:
                    orch.run()
                except Exception as e:  # 未捕获异常也入事件流，前端可见
                    self.events.append(task_id, "task.failed",
                                       {"error": f"{type(e).__name__}: {e}"})

            th = threading.Thread(target=run, daemon=True, name=f"task-{task_id}")
            self._threads[task_id] = th
            th.start()

    def resume_task(self, task_id: str) -> None:
        th = self._threads.get(task_id)
        if th is not None and th.is_alive():
            raise TaskRunningError(task_id)
        if self.task_status(task_id)["status"] == "done":
            raise NoCheckpointError("任务已完成，无需恢复")
        if self.events.load_checkpoint(task_id) is None:
            raise NoCheckpointError("任务无检查点，无法恢复")
        prompt = None
        for _seq, type_, payload in self.events.replay(task_id):
            if type_ == "task.created":
                prompt = payload.get("prompt")
        if prompt is None:
            raise NoCheckpointError("事件流中无 task.created，无法恢复")
        self.start_task(task_id, prompt)

    # ---- 状态 ----
    def task_status(self, task_id: str) -> dict:
        th = self._threads.get(task_id)
        running = th is not None and th.is_alive()
        events = self.events.replay(task_id)
        types = {t for _s, t, _p in events}
        # 子任务的三种终态事件都算“已完成”（正常完成 / 步数上限收尾 / 解析失败中止）；
        # subtask.paused 不算（未完成，可恢复）
        done = sum(1 for _s, t, _p in events
                   if t in ("subtask.finished", "subtask.max_steps", "subtask.aborted"))
        total: int | None = None
        for _s, t, payload in events:
            if t == "plan.created":
                total = len(payload.get("subtasks", []))
        if running:
            status = "running"
        elif "task.completed" in types:
            status = "done"
        elif "task.failed" in types:
            status = "failed"
        elif "task.paused" in types:
            status = "paused"
        elif events:
            status = "interrupted"  # 有事件但无终态（进程被杀等）
        else:
            status = "unknown"
        resumable = (status in ("paused", "failed", "interrupted")
                     and self.events.load_checkpoint(task_id) is not None)
        return {
            "task_id": task_id, "status": status, "resumable": resumable,
            "subtasks_done": done, "subtasks_total": total,
            "has_report": (self.storage / "reports" / f"{task_id}.md").exists(),
        }

    def list_tasks(self) -> list[dict]:
        return [self.task_status(tid) for tid in self.events.list_task_ids()]

    def report_text(self, task_id: str) -> str | None:
        f = self.storage / "reports" / f"{task_id}.md"
        return f.read_text(encoding="utf-8") if f.exists() else None


def _make_handler(manager: TaskManager) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "research-employee"

        def log_message(self, fmt, *args):  # 精简日志
            print(f"[web] {self.address_string()} {fmt % args}")

        # ---- 响应辅助 ----
        def _send(self, code: int, body: bytes, content_type: str,
                  no_cache: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if no_cache:
                # 前端单页无独立静态资源：禁用浏览器缓存，
                # 避免升级 JS 后用户仍跑到启发式缓存里的旧版页面
                self.send_header("Cache-Control", "no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
            self.end_headers()
            if not getattr(self, "_head_only", False):
                self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _text(self, text: str, content_type: str, code: int = 200,
                  no_cache: bool = False) -> None:
            self._send(code, text.encode("utf-8"), content_type, no_cache=no_cache)

        # ---- GET ----
        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            path, qs = parts.path, parse_qs(parts.query)
            if path in ("/", "/index.html"):
                self._text(manager.index_html, "text/html; charset=utf-8",
                          no_cache=True)
            elif path == "/api/status":
                self._json({"llm": manager.llm_desc,
                            "web": getattr(manager.web, "name", type(manager.web).__name__),
                            "storage": str(manager.storage)})
            elif path == "/api/tasks":
                self._json({"tasks": manager.list_tasks()})
            elif m := re.fullmatch(r"/api/tasks/([^/]+)/timeline", path):
                since = int((qs.get("since") or ["0"])[0])
                tid = m.group(1)
                events = [[seq, t, p] for seq, t, p in
                          manager.events.replay(tid, since_seq=since)]
                self._json({"events": events, "status": manager.task_status(tid)["status"]})
            elif m := re.fullmatch(r"/api/tasks/([^/]+)/report", path):
                text = manager.report_text(m.group(1))
                if text is None:
                    self._json({"error": "报告尚未生成"}, 404)
                else:
                    self._text(text, "text/markdown; charset=utf-8")
            else:
                self._json({"error": "not found"}, 404)

        def do_HEAD(self) -> None:
            # 与 GET 同路，不写正文（兼容浏览器/探活器的 HEAD 请求）
            self._head_only = True
            try:
                self.do_GET()
            finally:
                self._head_only = False

        # ---- POST ----
        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except json.JSONDecodeError:
                self._json({"error": "invalid json"}, 400)
                return
            if path == "/api/tasks":
                prompt = (body.get("prompt") or "").strip()
                if not prompt:
                    self._json({"error": "prompt 不能为空"}, 400)
                    return
                tid = (body.get("task_id") or ("t-" + uuid.uuid4().hex[:8])).strip()
                limit = int(body.get("limit_subtasks") or 3)
                try:
                    manager.start_task(tid, prompt, limit)
                except TaskRunningError:
                    self._json({"error": "任务正在运行中", "task_id": tid}, 409)
                    return
                self._json({"task_id": tid, "status": "running"}, 202)
            elif m := re.fullmatch(r"/api/tasks/([^/]+)/resume", path):
                try:
                    manager.resume_task(m.group(1))
                except TaskRunningError:
                    self._json({"error": "任务正在运行中"}, 409)
                    return
                except NoCheckpointError as e:
                    self._json({"error": str(e)}, 400)
                    return
                self._json({"task_id": m.group(1), "status": "running"}, 202)
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def create_server(storage_root: str | Path, host: str = "127.0.0.1",
                  port: int = 8000, llm=None, web=None, verbose: bool = False):
    """构造 (server, manager)。port=0 取临时端口（测试用）。"""
    manager = TaskManager(storage_root, llm=llm, web=web, verbose=verbose)
    server = ThreadingHTTPServer((host, port), _make_handler(manager))
    return server, manager
