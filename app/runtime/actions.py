"""结构化动作协议：LLM 输出必须是单个 JSON 动作。

{action: <tool|finish_subtask>, args: {...}}
解析失败不直接终止循环：把错误作为反馈返回给 LLM 重试（见 agent_loop）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

VALID_ACTIONS = {"web_search", "web_fetch", "read_artifact", "write_artifact",
                 "list_artifacts", "finish_subtask"}


class ActionParseError(Exception):
    pass


@dataclass
class Action:
    name: str
    args: dict = field(default_factory=dict)


def parse_action(raw: str) -> Action:
    """从 LLM 输出中提取第一个 JSON 对象并校验。"""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ActionParseError(f"未找到 JSON 动作块: {raw[:120]!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ActionParseError(f"JSON 解析失败: {e}")
    name = obj.get("action")
    if name not in VALID_ACTIONS:
        raise ActionParseError(f"未知动作 {name!r}，允许: {sorted(VALID_ACTIONS)}")
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        raise ActionParseError("args 必须是对象")
    return Action(name, args)
