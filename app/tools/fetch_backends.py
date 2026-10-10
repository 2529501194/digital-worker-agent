"""网页抓取后端（M2）：httpx 抓取 + HTML→Markdown + 磁盘缓存（幂等重放基础）。

- HTML→Markdown 用标准库 html.parser 实现（无第三方依赖）；
- 磁盘缓存：sha1(url).json，TTL 内直接复用 → 崩溃恢复重放时不重复抓取；
- 大小上限防单页灌爆上下文；
- http_client / cache_dir 可注入 → 离线测试（本地 HTTP server 验证真实链路）。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import httpx

MAX_CONTENT_CHARS = 200_000
UA = "ResearchDigitalEmployee/0.2 (research-agent; +https://example.local/bot)"


# ---------------------------------------------------------------- HTML→MD

class _MarkdownExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template"}
    _HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ",
                 "h5": "##### ", "h6": "###### "}
    _BLOCK_END = {"p", "div", "section", "article", "ul", "ol", "table",
                  "pre", "blockquote", "header", "footer", "main"}
    _BLOCK_START = {"p", "div", "br", "hr", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._lines: list[str] = []
        self._cur: list[str] = []
        self._skip_depth = 0
        self._heading_prefix = ""

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._HEADINGS:
            self._flush()
            self._heading_prefix = self._HEADINGS[tag]
        elif tag == "li":
            self._flush()
            self._heading_prefix = "- "
        elif tag in self._BLOCK_START:
            self._flush()

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK_END or tag in self._HEADINGS:
            self._flush()
            self._heading_prefix = ""

    def handle_data(self, data):
        if self._skip_depth or not data.strip():
            return
        self._cur.append(data)  # 保留原始空白，flush 时统一折叠

    def _flush(self):
        if self._cur:
            text = re.sub(r"\s+", " ", "".join(self._cur)).strip()
            if text:
                self._lines.append(self._heading_prefix + text)
            self._cur = []

    def markdown(self) -> str:
        self._flush()
        out, prev_blank = [], True
        for line in self._lines:
            if not line.strip():
                continue
            if not prev_blank:
                out.append("")
            out.append(line)
            prev_blank = False
        return "\n".join(out).strip() + "\n"


def html_to_markdown(html: str) -> str:
    parser = _MarkdownExtractor()
    parser.feed(html)
    return parser.markdown()


# ---------------------------------------------------------------- 抓取后端

class FetchBackend:
    """接口：fetch(url) -> (markdown, error)。子类需维护 fetch_counts。"""

    def __init__(self):
        self.fetch_counts: dict[str, int] = {}

    def fetch(self, url: str) -> tuple[str | None, str | None]:
        raise NotImplementedError


class HttpxFetchBackend(FetchBackend):
    def __init__(self, cache_dir: str | Path | None = None,
                 ttl_seconds: int = 6 * 3600,
                 http_client: httpx.Client | None = None,
                 max_chars: int = MAX_CONTENT_CHARS):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._http = http_client or httpx.Client(
            timeout=30.0, follow_redirects=True)
        self.max_chars = max_chars

    # ---- 缓存 ----
    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    def _cache_get(self, url: str) -> str | None:
        p = self._cache_path(url)
        if p and p.exists():
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                if time.time() - rec.get("fetched_at", 0) < self.ttl_seconds:
                    return rec["markdown"]
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def _cache_put(self, url: str, markdown: str) -> None:
        p = self._cache_path(url)
        if p:
            p.write_text(json.dumps({"url": url, "markdown": markdown,
                                     "fetched_at": time.time()},
                                    ensure_ascii=False), encoding="utf-8")

    # ---- 抓取 ----
    def fetch(self, url: str) -> tuple[str | None, str | None]:
        self.fetch_counts[url] = self.fetch_counts.get(url, 0) + 1

        cached = self._cache_get(url)
        if cached is not None:
            return cached, None

        try:
            resp = self._http.get(url, headers={"User-Agent": UA})
        except httpx.HTTPError as e:
            return None, f"网络错误: {e.__class__.__name__}: {e}"
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"

        ctype = resp.headers.get("content-type", "").lower()
        text = resp.text
        if "html" in ctype or text.lstrip()[:1].lower() == "<":
            text = html_to_markdown(text)
        text = text[: self.max_chars]
        if not text.strip():
            return None, "页面内容为空"
        self._cache_put(url, text)
        return text, None
