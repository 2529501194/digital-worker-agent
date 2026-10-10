"""You.com 免费 MCP 搜索/抓取后端（无需 API key，按 IP 限流）。

端点: https://api.you.com/mcp?profile=free
协议: MCP streamable HTTP（JSON-RPC 2.0，SSE 响应）
工具: you-search(query, count, extraction="full_page")
      返回 results.web[]: url/title/description/snippets + contents.markdown（整页正文）

设计要点:
- 一次搜索调用同时带回结果列表与整页 Markdown → fetch 优先复用最近一次
  搜索中已取到的正文，避免额外的限流调用；未命中才用"URL 作查询词"兜底；
- fetch_cache / fetch_counts 语义与其他后端一致：崩溃恢复重放不重复抓取；
- HTTP 层可注入（self.http.post(...)），离线测试不发真实请求。
"""
from __future__ import annotations

import json
import re

DEFAULT_URL = "https://api.you.com/mcp?profile=free"
_MAX_MD_CHARS = 20_000   # 单页正文截断，防上下文爆炸


class YouMCPClient:
    """MCP streamable HTTP 最小客户端（仅本工具链需要的 tools/call）。"""

    def __init__(self, http=None, base_url: str = DEFAULT_URL, timeout: float = 180.0):
        import httpx
        self.http = http or httpx
        self.base_url = base_url
        self.timeout = timeout
        self._id = 0

    def call_tool(self, name: str, arguments: dict) -> dict:
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id,
                   "method": "tools/call",
                   "params": {"name": name, "arguments": arguments}}
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        r = self.http.post(self.base_url, json=payload,
                           headers=headers, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"you-mcp: HTTP {r.status_code}: {r.text[:200]}")
        for msg in _parse_mcp_stream(r.text):
            res = msg.get("result")
            if isinstance(res, dict) and res.get("content"):
                text = res["content"][0].get("text", "")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"results": {"web": []},
                            "raw_text": text}
        raise RuntimeError("you-mcp: 响应中没有工具结果")


def _parse_mcp_stream(text: str) -> list[dict]:
    """SSE（data: 行）或裸 JSON 流 → JSON 消息列表。"""
    if "data:" in text:
        chunks = [l[5:].strip() for l in text.splitlines()
                  if l.startswith("data:")]
    else:
        chunks = [text]
    msgs: list[dict] = []
    dec = json.JSONDecoder()
    for chunk in chunks:
        i = 0
        while True:
            while i < len(chunk) and chunk[i] in " \r\n\t":
                i += 1
            if i >= len(chunk):
                break
            try:
                obj, i = dec.raw_decode(chunk, i)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                msgs.append(obj)
    return msgs


def _norm_md(item: dict) -> str:
    md = ((item.get("contents") or {}).get("markdown") or "").strip()
    if len(md) > _MAX_MD_CHARS:
        md = md[:_MAX_MD_CHARS] + "\n...[截断]"
    return md


class YouMCPWebBackend:
    """WebBackend 接口实现（search / fetch / fetch_counts / name）。"""

    name = "you-mcp"

    def __init__(self, client: YouMCPClient | None = None):
        self.client = client or YouMCPClient()
        self.fetch_cache: dict[str, str] = {}
        self.fetch_counts: dict[str, int] = {}
        self._md_by_url: dict[str, str] = {}   # 最近搜索带回的整页正文

    # ---- 搜索 ----
    def search(self, query: str, limit: int = 3) -> list[dict]:
        data = self.client.call_tool("you-search", {
            "query": query, "count": limit, "extraction": "full_page"})
        web = (data.get("results") or {}).get("web") or []
        hits: list[dict] = []
        for item in web[:limit]:
            url = item.get("url")
            if not url:
                continue
            desc = item.get("description") or ""
            if not desc and item.get("snippets"):
                desc = item["snippets"][0]
            hits.append({"url": url,
                         "title": item.get("title", ""),
                         "snippet": re.sub(r"\s+", " ", desc)[:300]})
            md = _norm_md(item)
            if md:
                self._md_by_url[url] = md
        return hits

    # ---- 抓取 ----
    def fetch(self, url: str) -> tuple[str | None, str | None]:
        """返回 (markdown, error)。复用搜索带回的正文；未命中用 URL 兜底搜索。"""
        self.fetch_counts[url] = self.fetch_counts.get(url, 0) + 1
        if url in self.fetch_cache:
            return self.fetch_cache[url], None
        if url in self._md_by_url:
            self.fetch_cache[url] = self._md_by_url[url]
            return self._md_by_url[url], None
        # 兜底：用 URL 本身做查询词，期望命中原页
        data = self.client.call_tool("you-search", {
            "query": url, "count": 5, "extraction": "full_page"})
        web = (data.get("results") or {}).get("web") or []
        for item in web:
            if item.get("url") == url and _norm_md(item):
                md = _norm_md(item)
                self._md_by_url[url] = md
                self.fetch_cache[url] = md
                return md, None
        return None, f"you-mcp: 未取到 {url} 的正文（搜索兜底也未命中）"
