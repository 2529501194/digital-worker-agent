"""真实 Web 链路测试：本地 HTTP server 验证 httpx 抓取/HTML→MD/缓存，
搜索 Provider 解析用 FakeHTTP 验证（不发真实网络请求）。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from app.tools.fetch_backends import HttpxFetchBackend, html_to_markdown
from app.tools.search_providers import BraveSearchProvider, SerpApiProvider
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
