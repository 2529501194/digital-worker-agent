"""模型网关（M1：接口 + Mock 实现）。

ModelProvider 接口是 M2 接真实模型的边界：
- M2：OpenAI 兼容实现（重试/退避/降级链/熔断，见设计方案 3.3）
- M1：MockResearchLLM 是"无状态规则化研究员"——只依据输入上下文里的
  STAGE/FOCUS/SCRATCH/ARTIFACTS 做决策，因此天然支持崩溃后从事件流恢复。
"""
from __future__ import annotations

import ast
import json
import re
from typing import Protocol

from .llm import CallMeta


class ModelProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


# ---------------------------------------------------------------- Mock LLM

PLAN_SUBTASKS = [
    {"title": "Claude Code 架构调研", "focus": "Claude Code agentic loop architecture"},
    {"title": "Codex CLI 架构调研", "focus": "Codex CLI architecture tools sandbox"},
    {"title": "Cursor Agent 架构调研", "focus": "Cursor agent architecture context"},
    {"title": "OpenHands 架构调研", "focus": "OpenHands agent event stream"},
]

_COMPARE_DIMS = ["任务规划", "工具使用", "上下文管理"]


def _line(ctx: str, key: str) -> str:
    m = re.search(rf"^{key}:\s*(.*)$", ctx, re.M)
    return m.group(1).strip() if m else ""


def _scratch_lines(ctx: str) -> list[str]:
    m = re.search(r"^SCRATCH:\n(.*?)(?:\n\w+:|\nARTIFACTS:|\Z)", ctx, re.M | re.S)
    if not m:
        return []
    return [l for l in m.group(1).splitlines() if l.strip() and l.strip() != "(empty)"]


class MockResearchLLM:
    """无状态规则化研究员：每个阶段只做"下一步该做什么"的决策。"""

    name = "mock-researcher-v0"

    def complete(self, system: str, user: str) -> str:
        stage = _line(user, "STAGE")
        if stage == "PLAN":
            return self._plan(user)
        if stage == "WRITE":
            return self._write_report(user)
        raw = self._next_execute_action(user)
        self._set_meta(raw)
        return raw

    def _plan(self, ctx: str) -> str:
        limit_raw = _line(ctx, "PLAN_LIMIT")
        limit = int(limit_raw) if limit_raw else 0
        subtasks = PLAN_SUBTASKS[:limit] if limit else PLAN_SUBTASKS
        raw = json.dumps({"subtasks": subtasks}, ensure_ascii=False)
        self._set_meta(raw)
        return raw

    def _set_meta(self, raw: str) -> None:
        # mock 无真实 usage，按字符估算 token，保证预算/事件链路一致
        self.last_meta = CallMeta(provider=self.name, model=self.name,
                                  tokens_out=max(1, len(raw) // 3))

    # ---- EXECUTE：依据 scratch 决定下一步 ----
    def _next_execute_action(self, ctx: str) -> str:
        focus = _line(ctx, "FOCUS")
        title = _line(ctx, "SUBTASK")
        scratch = _scratch_lines(ctx)
        search_line = next((l for l in scratch if l.startswith('web_search "')), None)
        fetch_line = next((l for l in scratch if l.startswith("web_fetch ") and "=> OK" in l), None)
        wrote_evidence = any(l.startswith("write_artifact [evidence]") and "=>" in l and "ERROR" not in l
                            for l in scratch)

        if search_line is None:
            return json.dumps(
                {"action": "web_search", "args": {"query": focus}}, ensure_ascii=False)

        if fetch_line is None:
            m = re.search(r"\[1\]\s+(\S+)", search_line)
            url = m.group(1) if m else ""
            if not url:
                # 搜索无结果：直接收尾并说明 gap
                return json.dumps(
                    {"action": "finish_subtask",
                     "args": {"summary": f"未找到 {focus} 相关资料（搜索无结果）",
                              "gaps": ["资料缺失"]}}, ensure_ascii=False)
            return json.dumps(
                {"action": "web_fetch", "args": {"url": url}}, ensure_ascii=False)

        if not wrote_evidence:
            m = re.search(r"^web_fetch (\S+) => OK \| (.*)$", fetch_line, re.M)
            url, snippet = (m.group(1), m.group(2)) if m else ("", "")
            evidence = (
                f"## 证据摘录：{title}\n\n"
                f"来源：{url}\n\n"
                f"摘录：{snippet}\n\n"
                f"（M1 mock 摘录；真实实现中由 LLM 对抓取全文做带引用的摘取）"
            )
            return json.dumps(
                {"action": "write_artifact",
                 "args": {"type": "evidence", "title": f"{title} 证据摘录",
                          "content": evidence, "refs": [url]}}, ensure_ascii=False)

        return json.dumps(
            {"action": "finish_subtask",
             "args": {"summary": f"{title}：已完成搜索、抓取与证据摘录"}},
            ensure_ascii=False)

    # ---- WRITE：汇总报告 ----
    def _write_report(self, ctx: str) -> str:
        task = _line(ctx, "TASK")
        plan_titles = re.findall(r"^PLAN:\n((?:- .*\n)+)", ctx, re.M)
        titles = [l[2:].strip() for l in plan_titles[0].splitlines()] if plan_titles else []

        artifacts: list[dict] = []
        cur: dict | None = None
        buf: list[str] = []
        in_art = False

        def flush() -> None:
            nonlocal cur, buf
            if cur is not None:
                cur["content"] = " ".join(buf).strip()
                artifacts.append(cur)
            cur, buf = None, []

        for line in ctx.splitlines():
            if line == "ARTIFACTS:":
                in_art = True
                continue
            if not in_art:
                continue
            m = re.match(r"^- (\S+) \[(\w+)\] (.+?) refs=(\[.*\])$", line)
            if m:
                flush()
                refs_raw = m.group(4)
                try:
                    refs = json.loads(refs_raw)
                except json.JSONDecodeError:
                    refs = ast.literal_eval(refs_raw)  # Python repr 容错
                cur = {"id": m.group(1), "title": m.group(3), "refs": refs}
            elif line.startswith("  "):
                buf.append(line.strip())
            elif not line.strip():
                continue
            else:
                in_art = False  # 遇到新的顶层 section，工件块结束
        flush()

        out = [f"# 研究报告：{task}", ""]
        out += ["## 概述",
                f"本报告由研究数字员工自动生成（M1 mock 流水线），覆盖 {len(titles)} 个研究对象，"
                f"比较维度：{' / '.join(_COMPARE_DIMS)}。", ""]
        for t in titles:
            evs = [a for a in artifacts if t.split(" ")[0] in a["title"]
                   or a["title"].startswith(t)]
            out += [f"## {t}"]
            if evs:
                for a in evs:
                    content = a.get("content", "").replace("\n", " ")[:200]
                    out.append(f"- 证据 {a['id']}：{content}")
                    out.append(f"  - 引用：{', '.join(a.get('refs', []))}")
            else:
                out.append("- （该对象证据缺失，见未完成项）")
            out.append("")
        refs = sorted({r for a in artifacts for r in a.get("refs", [])})
        out += ["## 参考文献"]
        out += [f"- {r}" for r in refs]
        out += ["", "> M1 说明：报告由 mock LLM 汇总；M2 接入真实模型后由 LLM 分节写作并挂引用。"]
        return "\n".join(out)


class ScriptedLLM:
    """按脚本顺序返回固定响应 —— 用于 agent_loop 单测（含解析失败恢复）。"""

    name = "scripted-llm"

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[str] = []
        self.last_meta: CallMeta | None = None

    def complete(self, system: str, user: str) -> str:
        self.calls.append(user)
        if not self.responses:
            raise RuntimeError("ScriptedLLM 脚本耗尽")
        raw = self.responses.pop(0)
        self.last_meta = CallMeta(provider=self.name, model=self.name,
                                  tokens_out=max(1, len(raw) // 3))
        return raw
