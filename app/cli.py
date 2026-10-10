"""CLI：提交任务 / 恢复任务 / 查看事件时间线。

    python -m app.cli run --prompt "…" [--task-id t-001] [--storage ./storage]
                          [--limit-subtasks 3]
    python -m app.cli resume --task-id t-001 --storage ./storage
    python -m app.cli timeline --task-id t-001 --storage ./storage

后端选择（见 app/config.py）：
- LLM：RE_EMP_LLM_API_KEY / OPENAI_API_KEY（+ 可选 RE_EMP_LLM_FALLBACK_* 降级链）
- Web：BRAVE_API_KEY / SERPAPI_API_KEY
- 均未配置 → mock 离线演示模式（打印告警）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from .config import AppConfig, build_llm, build_web
from .runtime.budget import Budget
from .runtime.events import EventStore
from .runtime.orchestrator import Orchestrator
from .tools.artifact_store import ArtifactStore
from .tools.registry import RESEARCHER_WHITELIST, ToolRouter

# Windows 控制台默认 GBK，强制 UTF-8 输出避免中文乱码
if sys.platform == "win" and sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _build_env(storage: Path, limit_subtasks: int | None,
               verbose: bool = True):
    cfg = AppConfig.from_env()
    events = EventStore(storage / "events.db")
    artifacts = ArtifactStore(storage / "artifacts")
    llm = build_llm(cfg, verbose=verbose)
    web = build_web(cfg, storage, verbose=verbose)
    tools = ToolRouter(web, artifacts, RESEARCHER_WHITELIST)
    return events, artifacts, tools, llm, limit_subtasks


def _run(task_id: str, prompt: str, storage: Path, limit_subtasks: int | None) -> None:
    events, artifacts, tools, llm, limit = _build_env(storage, limit_subtasks)
    orch = Orchestrator(
        task_id=task_id, task_prompt=prompt, llm=llm, tools=tools,
        events=events, artifacts=artifacts, budget=Budget(),
        report_dir=storage / "reports", plan_limit=limit)
    t0 = time.time()
    outcome = orch.run()
    print(f"[task {task_id}] status={outcome.status} "
          f"subtasks={outcome.completed_subtasks}/{outcome.total_subtasks} "
          f"({time.time() - t0:.1f}s)")
    if outcome.report_path:
        print(f"report: {outcome.report_path}")
    for g in outcome.gaps:
        print(f"gap: {g}")
    if outcome.status == "done":
        print(f"提示：用 `python -m app.cli timeline --task-id {task_id} "
              f"--storage {storage}` 查看执行时间线")


def cmd_run(a: argparse.Namespace) -> None:
    storage = Path(a.storage)
    task_id = a.task_id or "t-" + uuid.uuid4().hex[:8]
    _run(task_id, a.prompt, storage, a.limit_subtasks)
    print(f"task_id: {task_id}")


def cmd_resume(a: argparse.Namespace) -> None:
    storage = Path(a.storage)
    events, artifacts, tools, llm, _limit = _build_env(storage, None, verbose=False)
    cp = events.load_checkpoint(a.task_id)
    if cp is None:
        print(f"任务 {a.task_id} 无检查点，无法恢复")
        sys.exit(1)
    prompt = None
    for _s, t, p in events.replay(a.task_id):
        if t == "task.created":
            prompt = p.get("prompt")
    if prompt is None:
        print("事件流中无 task.created，无法恢复 prompt")
        sys.exit(1)
    print(f"resuming {a.task_id} from stage={cp.stage} cursor={cp.subtask_cursor}")
    _run(a.task_id, prompt, storage, None)


def cmd_timeline(a: argparse.Namespace) -> None:
    storage = Path(a.storage)
    events, *_ = _build_env(storage, None, verbose=False)
    print(f"── 任务 {a.task_id} 事件时间线 ──")
    for seq, type_, payload in events.replay(a.task_id):
        extra = ""
        if type_ == "llm.called":
            extra = (f" stage={payload.get('stage')} subtask={payload.get('subtask')} "
                     f"provider={payload.get('provider')} "
                     f"tokens={payload.get('tokens_in', 0)}+{payload.get('tokens_out', 0)} "
                     f"({payload.get('latency_ms', 0)}ms, retry={payload.get('retries', 0)})")
        elif type_ == "tool.called":
            extra = f" {payload.get('tool')} ok={payload.get('ok')}"
            if payload.get("error"):
                extra += f" err={payload['error']}"
        elif type_ in ("subtask.finished", "plan.created", "report.generated",
                       "task.paused", "task.completed", "task.resumed",
                       "llm.parse_error", "checkpoint.saved"):
            extra = " " + json.dumps(payload, ensure_ascii=False)[:160]
        print(f"  #{seq:<3} {type_:<20}{extra}")


def cmd_serve(a: argparse.Namespace) -> None:
    from .web.server import create_server
    server, _ = create_server(a.storage, host=a.host, port=a.port, verbose=True)
    print(f"[web] http://{a.host}:{server.server_address[1]}  （Ctrl+C 停止）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[web] stopped")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="research-employee")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="提交并运行研究任务")
    pr.add_argument("--prompt", required=True)
    pr.add_argument("--task-id")
    pr.add_argument("--storage", default="storage")
    pr.add_argument("--limit-subtasks", type=int, default=3,
                    help="研究计划子任务上限（缩小版任务，默认 3）")
    pr.set_defaults(fn=cmd_run)

    pr = sub.add_parser("resume", help="从检查点恢复任务")
    pr.add_argument("--task-id", required=True)
    pr.add_argument("--storage", default="storage")
    pr.set_defaults(fn=cmd_resume)

    pr = sub.add_parser("timeline", help="打印事件时间线")
    pr.add_argument("--task-id", required=True)
    pr.add_argument("--storage", default="storage")
    pr.set_defaults(fn=cmd_timeline)

    pr = sub.add_parser("serve", help="启动 Web 前端（http://127.0.0.1:8000）")
    pr.add_argument("--storage", default="storage")
    pr.add_argument("--host", default="127.0.0.1")
    pr.add_argument("--port", type=int, default=8000)
    pr.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
