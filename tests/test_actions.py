"""parse_action / first_json_object：JSON 提取健壮性（贪婪正则缺陷的回归）。

旧实现 re.search(r"\\{.*\\}", re.S) 从第一个 { 匹配到最后一个 }：
LLM 输出多个 JSON 对象或后随含 } 杂文时过度匹配 → "Extra data" 解析失败
（真实案例：GLM-5.3 事件流中两次 llm.parse_error "Extra data: line 1 column 1171"）。
"""
from __future__ import annotations

import pytest

from app.runtime.actions import (ActionParseError, extract_first_json_object,
                                first_json_object, parse_action)


class TestExtractFirstJsonObject:

    def test_plain_object(self):
        assert extract_first_json_object('{"a":1}') == '{"a":1}'

    def test_stops_at_first_complete_object_not_last_brace(self):
        raw = '{"a":1} {"b":2} 后文 } 还有}'
        assert extract_first_json_object(raw) == '{"a":1}'

    def test_nested_objects(self):
        assert extract_first_json_object('{"a":{"b":1}}') == '{"a":{"b":1}}'

    def test_braces_inside_strings_ignored(self):
        raw = '{"q":"} { } {"}'
        assert extract_first_json_object(raw) == raw

    def test_escaped_quotes_in_strings(self):
        raw = r'{"q":"he said \"} { \""}'
        assert extract_first_json_object(raw) == raw

    def test_unbalanced_tail_returns_none(self):
        assert extract_first_json_object('{"a":1') is None
        assert extract_first_json_object('no braces') is None

    def test_start_offset(self):
        raw = '前文 {x} {"a":1}'
        # 从第二个 { 开始提取
        assert extract_first_json_object(raw, raw.find('{"a"')) == '{"a":1}'


class TestFirstJsonObject:

    def test_skips_unparseable_placeholder(self):
        raw = '提示 {x} 如下 {"action":"web_search","args":{}}'
        assert first_json_object(raw) == {"action": "web_search", "args": {}}

    def test_first_of_two_objects_wins(self):
        raw = '{"subtasks": [{"goal": "a"}]} {"compare_dimensions": ["x"]}'
        assert first_json_object(raw) == {"subtasks": [{"goal": "a"}]}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="未找到"):
            first_json_object("plain text only")

    def test_all_candidates_invalid_raises(self):
        # 每个候选都平衡但都不是 JSON → 耗尽后报“JSON 解析失败”
        with pytest.raises(ValueError, match="JSON 解析失败"):
            first_json_object("{bad} {worse}")
        # 最后一个候选不平衡 → 报“未找到”
        with pytest.raises(ValueError, match="未找到"):
            first_json_object("{bad} {worse")


class TestParseAction:

    def test_plain(self):
        a = parse_action('{"action":"web_search","args":{"query":"x"}}')
        assert a.name == "web_search" and a.args == {"query": "x"}

    def test_surrounding_prose(self):
        a = parse_action('好的，动作如下：{"action":"web_fetch","args":{"url":"https://a.b"}} 完成。')
        assert a.name == "web_fetch"

    def test_multiple_json_objects_picks_first(self):
        """旧贪婪正则在此场景必然失败（跨两个对象匹配 → Extra data）。"""
        raw = ('{"action":"web_search","args":{"query":"q"}} '
               '{"action":"finish_subtask","args":{"summary":"done"}}')
        assert parse_action(raw).name == "web_search"

    def test_trailing_prose_with_braces(self):
        a = parse_action(
            '{"action":"finish_subtask","args":{"summary":"s"}} 注意：{说明} 结束')
        assert a.name == "finish_subtask"

    def test_args_missing_defaults_empty(self):
        assert parse_action('{"action":"list_artifacts"}').args == {}

    def test_unknown_action_rejected(self):
        with pytest.raises(ActionParseError, match="未知动作"):
            parse_action('{"action":"fly","args":{}}')

    def test_args_must_be_object(self):
        with pytest.raises(ActionParseError, match="args 必须是对象"):
            parse_action('{"action":"web_search","args":[1,2]}')

    def test_no_json_raises(self):
        with pytest.raises(ActionParseError, match="未找到"):
            parse_action("no json here")

    def test_unbalanced_raises(self):
        with pytest.raises(ActionParseError, match="未找到"):
            parse_action('{"action":"web_search"')
