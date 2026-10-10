"""上下文管理器（M1 最小版）。

把当前阶段、子任务、scratchpad（循环内记忆）、工件摘要组装成 LLM 输入。
M2 加滚动摘要压缩；M3 加上下文预算触发压缩事件。
"""
from __future__ import annotations

import json
from typing import Any


def build_context(
    stage: str,
    task: str,
    subtask: dict | None = None,
    plan: list[dict] | None = None,
    scratch: list[str] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    content_chars: int = 0,
    plan_limit: int | None = None,
) -> str:
    lines = [f"STAGE: {stage}", f"TASK: {task}"]
    if stage == "PLAN" and plan_limit:
        lines.append(f"PLAN_LIMIT: {plan_limit}")
    lines.append("")
    if stage == "EXECUTE" and subtask is not None:
        lines += [f"SUBTASK: {subtask.get('title','')}",
                  f"FOCUS: {subtask.get('focus','')}",
                  "SCRATCH:"]
        lines += list(scratch or []) or ["(empty)"]
        lines.append("")

    if stage == "WRITE":
        if plan:
            lines.append("PLAN:")
            lines += [f"- {s.get('title','')}" for s in plan]
            lines.append("")
        lines.append("ARTIFACTS:")
        for a in artifacts or []:
            content = a.get("content", "")
            if content_chars > 0 and len(content) > content_chars:
                content = content[:content_chars] + " ...[截断]"
            lines.append(f"- {a['id']} [{a['type']}] {a['title']} "
                         f"refs={json.dumps(a.get('refs', []), ensure_ascii=False)}")
            lines.append("  " + content.replace("\n", "\n  "))
        lines.append("")

    return "\n".join(lines)
