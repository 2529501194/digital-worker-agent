"""真实 Web 链路测试：本地 HTTP server 验证 httpx 抓取/HTML→MD/缓存，
搜索 Provider 解析用 FakeHTTP 验证（不发真实网络请求）。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from app.tools.fetch_backends import HttpxFetchBackend, html_to_markdown
from app.tools.search_providers import (BraveSearchProvider, LangSearchProvider,
                                       SerpApiProvider)
from app.tools.registry import ToolRouter
from app.tools.artifact_store import ArtifactStore


# ---------------------------------------------------------------- HTML→MD

def test_html_to_markdown_basics():
    html = """<html><head><style>x{}</style><script>bad()</script></head>
    <body><h1>Title Here</h1>
    <p>First paragraph with   spaces.</p>
    <ul><li>item one</li><li>item two</li></ul>
    <h2>Section</h2><p>After section.</p></body></html>"""
    md = html_to_markdown(html)
    assert "# Title Here" in md
    assert "First paragraph with spaces." in md
    assert "- item one" in md and "- item two" in md
    assert "## Section" in md
    assert "bad()" not in md and "x{}" not in md  # script/style 被剥离


# ---------------------------------------------------------------- 本地 HTTP server

class _CountingHandler(BaseHTTPRequestHandler):
    hits: dict[str, int] = {}

    def do_GET(self):
        _CountingHandler.hits[self.path] = \
            _CountingHandler.hits.get(self.path, 0) + 1
        if self.path == "/page":
            body = ("<html><body><h1>Real Page</h1>"
                    "<p>Content about <b>agent loops</b>.</p>"
                    "<ul><li>plan</li><li>act</li><li>observe</li></ul>"
                    "</body></html>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/plain":
            body = b"# plain markdown note\nsome text"
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture()
def http_server():
    _CountingHandler.hits = {}
    server = HTTPServer(("127.0.0.1", 0), _CountingHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base
    server.shutdown()


def test_httpx_fetch_html_to_markdown(http_server, tmp_path):
    be = HttpxFetchBackend(cache_dir=tmp_path / "cache")
    md, err = be.fetch(f"{http_server}/page")
    assert err is None
    assert "# Real Page" in md
    assert "Content about agent loops." in md
    assert "- plan" in md


def test_httpx_fetch_plain_markdown_passthrough(http_server, tmp_path):
    be = HttpxFetchBackend(cache_dir=tmp_path / "cache")
    md, err = be.fetch(f"{http_server}/plain")
    assert err is None and md.startswith("# plain markdown note")


def test_httpx_fetch_404(http_server, tmp_path):
    be = HttpxFetchBackend(cache_dir=tmp_path / "cache")
    md, err = be.fetch(f"{http_server}/nope")
    assert md is None and "404" in err


def test_fetch_disk_cache_prevents_refetch(http_server, tmp_path):
    cache = tmp_path / "cache"
    be = HttpxFetchBackend(cache_dir=cache)
    url = f"{http_server}/page"
    be.fetch(url)
    assert _CountingHandler.hits["/page"] == 1
    # 模拟"进程重启"：新实例、同一缓存目录
    be2 = HttpxFetchBackend(cache_dir=cache)
    md, err = be2.fetch(url)
    assert err is None and "# Real Page" in md
    assert _CountingHandler.hits["/page"] == 1  # 服务端只被打过一次
    # 计数仍记录（审计用）
    assert be2.fetch_counts[url] == 1


def test_fetch_network_error_graceful(tmp_path):
    be = HttpxFetchBackend(cache_dir=tmp_path / "cache",
                           http_client=httpx.Client(timeout=0.2))
    md, err = be.fetch("http://127.0.0.1:1/unreachable")
    assert md is None and err is not None and "网络错误" in err


# ---------------------------------------------------------------- 搜索 Provider

class FakeHTTP:
    def __init__(self, payload, status=200):
        self._payload, self._status = payload, status

    def get(self, url, params=None, headers=None, **kw):
        class R:
            status_code = self._status
            text = json.dumps(self._payload)
            def json(self_inner):
                return self._payload
        self.last_params = params
        return R()


def test_brave_provider_parsing():
    payload = {"web": {"results": [
        {"url": "https://a.example", "title": "A", "description": "desc-a"},
        {"url": None, "title": "skip-me"},
        {"url": "https://b.example", "title": "B"},
    ]}}
    p = BraveSearchProvider("key", http_client=FakeHTTP(payload))
    out = p.search("q", limit=3)
    assert [r["url"] for r in out] == ["https://a.example", "https://b.example"]
    assert out[0]["snippet"] == "desc-a"


def test_serpapi_provider_parsing():
    payload = {"organic_results": [
        {"link": "https://c.example", "title": "C", "snippet": "snip"},
    ]}
    p = SerpApiProvider("key", http_client=FakeHTTP(payload))
    out = p.search("q")
    assert out == [{"url": "https://c.example", "title": "C", "snippet": "snip"}]


_json_dumps = json.dumps  # post() 的 json 参数会遮蔽模块名，模块级先绑定


class FakePostHTTP:
    """LangSearch 是 POST + Bearer 认证，单独一个桩记录请求。"""

    def __init__(self, payload, status=200):
        self._payload, self._status = payload, status
        self.requests = []

    def post(self, url, json=None, headers=None, **kw):
        self.requests.append({"url": url, "json": json, "headers": headers})

        class R:
            status_code = self._status
            text = _json_dumps(self._payload)
            def json(self_inner):
                return self._payload
        return R()


def test_langsearch_provider_parsing():
    payload = {
        "code": 200, "log_id": "req-1", "msg": None,
        "data": {"_type": "SearchResponse", "webPages": {"value": [
            {"name": "A", "url": "https://a.example", "snippet": "snip-a",
             "datePublished": "2026-01-01"},
            {"name": None, "url": "https://b.example"},
            {"name": "C", "url": None},
        ]}},
        "usage": {"input_tokens": 5, "output_tokens": 15},
    }
    fake = FakePostHTTP(payload)
    out = LangSearchProvider("key", http_client=fake).search("q", limit=3)
    # url 为 None 的条目跳过；name 缺失 → title 空串
    assert [r["url"] for r in out] == ["https://a.example", "https://b.example"]
    assert out[0] == {"url": "https://a.example", "title": "A", "snippet": "snip-a"}
    assert out[1] == {"url": "https://b.example", "title": "", "snippet": ""}
    # 请求契约：POST JSON + Bearer 认证
    req = fake.requests[0]
    assert req["json"] == {"query": "q", "count": 3}
    assert req["headers"]["Authorization"] == "Bearer key"


def test_langsearch_text_mode_snippet_fallback():
    """text 模式（snippet 缺省、text 存在）时用 text 前 200 字兜底。"""
    payload = {"data": {"webPages": {"value": [
        {"name": "A", "url": "https://a.example",
         "text": "x" * 500},
    ]}}}
    out = LangSearchProvider("key", http_client=FakePostHTTP(payload)).search("q")
    assert len(out[0]["snippet"]) == 200


def test_langsearch_count_clamped_to_api_range():
    fake = FakePostHTTP({"data": {"webPages": {"value": []}}})
    LangSearchProvider("key", http_client=fake).search("q", limit=999)
    assert fake.requests[0]["json"]["count"] == 50
    LangSearchProvider("key", http_client=fake).search("q", limit=0)
    assert fake.requests[1]["json"]["count"] == 1


def test_langsearch_error_raises_observable():
    """429/401 等错误 → RuntimeError（ToolRouter 捕获后作为 observation 交回 LLM）。"""
    fake = FakePostHTTP({"code": "429", "log_id": "x",
                         "message": "daily allowance exceeded"}, status=429)
    p = LangSearchProvider("key", http_client=fake)
    with pytest.raises(RuntimeError, match=r"langsearch: HTTP 429"):
        p.search("q")


class _FailingSearch:
    name = "broken"
    def search(self, query, limit=3):
        raise RuntimeError("api 502")


def test_search_failure_returns_observable_error(tmp_path):
    """搜索 API 挂掉 → 工具层返回 ok=False + 可观察 scratch，循环不终止。"""
    from app.tools.web import ToolResult
    router = ToolRouter(_FailingWeb(), ArtifactStore(tmp_path))
    r = router.run("t1", "web_search", {"query": "x"})
    assert not r.ok and "api 502" in r.error
    assert "ERROR" in r.scratch


class _FailingWeb:
    """ToolRouter 需要的最小 web 后端接口。"""
    fetch_counts = {}

    def search(self, query, limit=3):
        raise RuntimeError("api 502")

    def fetch(self, url):
        return None, "not used"


# ---------------------------------------------------------------- build_web 选择

def test_build_web_prefers_langsearch(monkeypatch, tmp_path):
    from app.config import AppConfig, build_web
    monkeypatch.setenv("LANGSEARCH_API_KEY", "ls-key")
    monkeypatch.setenv("BRAVE_API_KEY", "brave-key")  # 优先级低于 LangSearch
    web = build_web(AppConfig.from_env(), tmp_path, verbose=False)
    assert web.name == "langsearch+http"


def test_build_web_falls_back_to_brave(monkeypatch, tmp_path):
    from app.config import AppConfig, build_web
    monkeypatch.delenv("LANGSEARCH_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "brave-key")
    web = build_web(AppConfig.from_env(), tmp_path, verbose=False)
    assert web.name == "brave+http"
