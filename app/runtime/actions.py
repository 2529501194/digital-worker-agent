"""结构化动作协议：LLM 输出必须是单个 JSON 动作。

{action: <tool|finish_subtask>, args: {...}}
解析失败不直接终止循环：把错误作为反馈返回给 LLM 重试（见 agent_loop）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

VALID_ACTIONS = {"web_search", "web_fetch", "read_artifact", "write_artifact",
                 "list_artifacts", "finish_subtask"}


class ActionParseError(Exception):
    pass


@dataclass
class Action:
    name: str
    args: dict = field(default_factory=dict)


def extract_first_json_object(raw: str, start: int = 0) -> str | None:
    """从 start 起第一个 { 开始，按括号平衡提取第一个完整 JSON 对象。

    字符串/转义感知：引号内的花括号（含 \" 转义）不参与配平。
    比贪婪正则 r"\\{.*\\}"（从第一个 { 匹配到最后一个 }）健壮：
    - 模型输出多个 JSON 对象 → 只取第一个完整对象，不过度匹配
    - 后随含 } 的杂文不会把范围拉爆（旧正则在此场景必然 JSON 解析失败）
    尾部不平衡（无右括号）→ None。
    """
    start = raw.find("{", start)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return None


def first_json_object(raw: str) -> dict:
    """依次尝试第一个可解析的平衡 JSON 块（供动作/计划解析共用）。

    第一个候选解析失败（如散文里的 {占位符}）时，从下一个 { 继续——
    容忍“杂文 + 多个 JSON”的模型输出。找不到可解析对象 → ValueError。
    """
    start = 0
    while True:
        block = extract_first_json_object(raw, start)
        if block is None:
            raise ValueError(f"未找到 JSON 块: {raw[:120]!r}")
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            nxt = raw.find("{", start + 1)
            if nxt == -1:
                raise ValueError(
                    f"JSON 解析失败: {raw[:120]!r} 中无有效 JSON 对象") from None
            start = nxt


def parse_action(raw: str) -> Action:
    """从 LLM 输出中提取第一个可解析 JSON 对象并校验。"""
    try:
        obj = first_json_object(raw)
    except ValueError as e:
        raise ActionParseError(str(e)) from e
    name = obj.get("action")
    if name not in VALID_ACTIONS:
        raise ActionParseError(f"未知动作 {name!r}，允许: {sorted(VALID_ACTIONS)}")
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        raise ActionParseError("args 必须是对象")
    return Action(name, args)
