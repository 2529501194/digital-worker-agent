"""You.com 免费 MCP 后端：SSE 解析 / 搜索映射 / fetch 复用与兜底（离线，fake http）。"""
import json

import pytest

from app.tools.mcp_you import YouMCPClient, YouMCPWebBackend, _parse_mcp_stream

# 按真实响应裁剪的样例：SSE 流里 2 条消息（logging 通知 + 工具结果）
def _sse_payload(web_items, with_sse=True):
    inner = json.dumps({"results": {"web": web_items}}, ensure_ascii=False)
    msgs = [
        {"jsonrpc": "2.0", "method": "notifications/message",
         "params": {"level": "info", "data": "ok"}},
        {"jsonrpc": "2.0", "id": 3, "result": {
            "content": [{"type": "text", "text": inner}],
            "isStructured": True}},
    ]
    body = "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs)
    if with_sse:
        body = "\n".join(f"data: {line}" for line in body.splitlines())
    return body


def _item(url, title, md=None):
    return {"url": url, "title": title, "description": "desc of " + title,
            "contents": {"markdown": md or ("page body of " + url)}}


class FakeHTTP:
    """记录调用序列，按脚本返回 (status, body)。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "args": json["params"]["arguments"]})
        status, body = self.script.pop(0) if len(self.script) > 1 \
            else self.script[0]
        return type("R", (), {"status_code": status, "text": body})()


def test_parse_sse_stream():
    body = _sse_payload([_item("https://a.example", "A")])
    msgs = _parse_mcp_stream(body)
    assert len(msgs) == 2 and msgs[1]["id"] == 3


def test_search_maps_hits_and_caches_markdown():
    body = _sse_payload([_item("https://a.example", "A", "A full page"),
                        _item("https://b.example", "B", "B full page")])
    client = YouMCPClient(http=FakeHTTP([(200, body)]))
    web = YouMCPWebBackend(client)
    hits = web.search("q", limit=2)
    assert [h["url"] for h in hits] == ["https://a.example", "https://b.example"]
    assert hits[0]["title"] == "A"

    # fetch 直接复用搜索带回的正文：不产生新的 HTTP 调用
    md, err = web.fetch("https://a.example")
    assert err is None and md == "A full page"
    assert web.fetch_counts == {"https://a.example": 1}


def test_fetch_fallback_searches_by_url():
    body1 = _sse_payload([_item("https://a.example", "A", "A page")])
    body2 = _sse_payload([_item("https://b.example", "B", "B page")])
    fake = FakeHTTP([(200, body1), (200, body2)])
    web = YouMCPWebBackend(YouMCPClient(http=fake))
    web.search("q")                       # 第一次：带回 a
    md, err = web.fetch("https://b.example")
    assert err is None and md == "B page"
    assert fake.calls[1]["args"]["query"] == "https://b.example"  # URL 兜底查询

    # 缓存命中：再 fetch 不新增调用
    web.fetch("https://b.example")
    assert len(fake.calls) == 2
    assert web.fetch_counts["https://b.example"] == 2


def test_fetch_unknown_url_returns_error():
    body = _sse_payload([_item("https://a.example", "A")])
    fake = FakeHTTP([(200, body)])
    web = YouMCPWebBackend(YouMCPClient(http=fake))
    md, err = web.fetch("https://missing.example")
    assert md is None and "未取到" in err


def test_http_error_raises():
    fake = FakeHTTP([(503, "backend down")])
    web = YouMCPWebBackend(YouMCPClient(http=fake))
    with pytest.raises(RuntimeError, match="503"):
        web.search("q")
