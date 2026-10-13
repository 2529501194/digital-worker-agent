"""搜索 Provider（M2）：LangSearch / Brave / SerpApi / Mock，统一 search(query, limit) 接口。

HTTP client 可注入 → 离线测试。失败抛异常，由 ToolRouter 捕获后
作为 observation 交回 LLM 决策（换关键词 / 换来源 / 记 gap）。
"""
from __future__ import annotations

import re
from typing import Protocol

import httpx

# 中文查询词 → 英文语料关键词（mock 语料库的简易跨语言对齐；真实搜索无需此步）
ZH_ALIAS = {
    "主流": "mainstream", "编码": "coding", "编程": "coding", "代码": "code",
    "架构": "architecture", "规划": "planning", "工具": "tools", "工具使用": "tools",
    "上下文": "context", "管理": "management", "比较": "compare", "对比": "compare",
    "列表": "list", "收集": "collect", "分析": "analysis", "差异": "differences",
}


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, limit: int = 3) -> list[dict]:
        """返回 [{url, title, snippet}]，按相关性排序。"""
        ...


class BraveSearchProvider:
    name = "brave"

    def __init__(self, api_key: str, http_client: httpx.Client | None = None,
                 base_url: str = "https://api.search.brave.com/res/v1/web/search",
                 timeout: float = 30.0):
        self.api_key = api_key
        self.base_url = base_url
        self._http = http_client or httpx.Client(timeout=timeout)

    def search(self, query: str, limit: int = 3) -> list[dict]:
        resp = self._http.get(
            self.base_url,
            params={"q": query, "count": str(limit), "summary": "true"},
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"brave: HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        out = []
        for item in (data.get("web") or {}).get("results", [])[:limit]:
            url = item.get("url")
            if not url:
                continue
            out.append({"url": url, "title": item.get("title", ""),
                        "snippet": item.get("description", "") or ""})
        return out


class LangSearchProvider:
    """LangSearch Web Search API（langsearch.com，免费每日 token 额度，5 RPS）。

    POST https://api.langsearch.com/v1/web-search
    Authorization: Bearer <LANGSEARCH_API_KEY>
    响应：{code, data: {webPages: {value: [{url, name, snippet, ...}]}}, usage}
    默认 snippet 模式（不请求整页文本）——整页由 web_fetch 负责，
    避免整页文本挤占 LLM 上下文与每日 token 额度。
    """

    name = "langsearch"

    def __init__(self, api_key: str, http_client: httpx.Client | None = None,
                 base_url: str = "https://api.langsearch.com/v1/web-search",
                 timeout: float = 30.0):
        self.api_key = api_key
        self.base_url = base_url
        self._http = http_client or httpx.Client(timeout=timeout)

    def search(self, query: str, limit: int = 3) -> list[dict]:
        count = max(1, min(int(limit), 50))  # API 允许 1-50
        resp = self._http.post(
            self.base_url,
            json={"query": query, "count": count},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"langsearch: HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        pages = ((data.get("data") or {}).get("webPages") or {}).get("value") or []
        out = []
        for item in pages[:limit]:
            url = item.get("url")
            if not url:
                continue
            snippet = item.get("snippet") or ""
            if not snippet and item.get("text"):  # text 模式下的兜底
                snippet = item["text"][:200]
            out.append({"url": url, "title": item.get("name") or "",
                        "snippet": snippet})
        return out


class SerpApiProvider:
    name = "serpapi"

    def __init__(self, api_key: str, http_client: httpx.Client | None = None,
                 base_url: str = "https://serpapi.com/search.json",
                 timeout: float = 30.0):
        self.api_key = api_key
        self.base_url = base_url
        self._http = http_client or httpx.Client(timeout=timeout)

    def search(self, query: str, limit: int = 3) -> list[dict]:
        resp = self._http.get(
            self.base_url,
            params={"engine": "google", "q": query, "num": str(limit),
                    "api_key": self.api_key},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"serpapi: HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        out = []
        for item in data.get("organic_results", [])[:limit]:
            url = item.get("link")
            if not url:
                continue
            out.append({"url": url, "title": item.get("title", ""),
                        "snippet": item.get("snippet", "") or ""})
        return out


class MockSearchProvider:
    """M1 的确定性 mock（从 MockWebBackend 抽出的打分逻辑）。"""

    name = "mock"

    def __init__(self, pages: dict[str, dict[str, str]]):
        self.pages = pages

    def search(self, query: str, limit: int = 3) -> list[dict]:
        words = [w for w in re.split(r"\s+", query.lower()) if len(w) > 2]
        scored: list[tuple[int, str]] = []
        for url, page in self.pages.items():
            title = page["title"].lower()
            content = page["content"].lower()
            score = 0
            for i, w in enumerate(words):
                # 中文词通过别名映射对齐英文语料；任一变体命中即计分
                variants = {w, ZH_ALIAS.get(w, "")}
                base = (2 if any(v in title for v in variants if v) else 0) \
                    + (1 if any(v in content for v in variants if v) else 0)
                score += base * (3 if i == 0 else 1)
            scored.append((score, url))
        scored.sort(key=lambda x: (-x[0], x[1]))
        results = []
        for score, url in scored[:limit]:
            if score == 0:
                break
            results.append({"url": url, "title": self.pages[url]["title"],
                            "snippet": self.pages[url]["content"][:100].replace("\n", " ")})
        return results
