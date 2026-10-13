"""配置装配（M2）：env 驱动，无 key 自动降级 mock 并告警。

LLM（OpenAI 兼容，可配主/备两级降级链）：
    RE_EMP_LLM_BASE_URL   默认 https://api.openai.com/v1
    RE_EMP_LLM_API_KEY
    RE_EMP_LLM_MODEL      默认 gpt-4o-mini
    RE_EMP_LLM_TIMEOUT / RE_EMP_LLM_MAX_RETRIES
    RE_EMP_LLM_FALLBACK_BASE_URL / _API_KEY / _MODEL   （可选降级模型）

Web：
    LANGSEARCH_API_KEY    → LangSearch 搜索（langsearch.com 免费 API，每日 token 额度）
    BRAVE_API_KEY         → Brave 搜索
    SERPAPI_API_KEY       → SerpApi 搜索
    RE_EMP_WEB=auto|http|mcp|mock
        auto（默认）：有 LangSearch/Brave/SerpApi key 用真实 http 后端
                      （优先级 LangSearch > Brave > SerpApi）；否则用 You.com 免费 MCP；
                     RE_EMP_YOU_MCP=off 可禁用 MCP，回到纯离线 mock
        mcp：强制 You.com 免费 MCP（api.you.com/mcp?profile=free，无需 key，IP 限流）
    RE_EMP_FETCH_TTL_SECONDS    抓取缓存 TTL（默认 6h）

存储：
    storage/  下 events.db / artifacts/ / reports/ / fetch_cache/
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .runtime.llm import ModelGateway, OpenAICompatibleProvider
from .runtime.model_gateway import MockResearchLLM, ModelProvider
from .tools.fetch_backends import HttpxFetchBackend
from .tools.web import MockWebBackend, RealWebBackend


@dataclass
class LLMProviderSpec:
    name: str
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0
    max_retries: int = 3


@dataclass
class AppConfig:
    llm_providers: list[LLMProviderSpec] = field(default_factory=list)
    web_mode: str = "auto"          # auto | http | mcp | mock
    langsearch_key: str | None = None
    brave_key: str | None = None
    serp_key: str | None = None
    you_mcp_enabled: bool = True    # 免费 MCP 搜索（无需 key，IP 限流）
    fetch_ttl_seconds: int = 6 * 3600
    fetch_cache_dir: str | Path | None = None

    @classmethod
    def from_env(cls) -> "AppConfig":
        cfg = cls()
        cfg.langsearch_key = os.environ.get("LANGSEARCH_API_KEY") or None
        cfg.brave_key = os.environ.get("BRAVE_API_KEY") or None
        cfg.serp_key = os.environ.get("SERPAPI_API_KEY") or None
        cfg.web_mode = os.environ.get("RE_EMP_WEB", "auto").lower()
        cfg.you_mcp_enabled = os.environ.get("RE_EMP_YOU_MCP", "on").lower() \
            not in ("off", "0", "false")
        cfg.fetch_ttl_seconds = int(os.environ.get("RE_EMP_FETCH_TTL_SECONDS",
                                                   6 * 3600))
        cfg.fetch_cache_dir = os.environ.get("RE_EMP_FETCH_CACHE_DIR") or None

        primary_key = os.environ.get("RE_EMP_LLM_API_KEY") or os.environ.get(
            "OPENAI_API_KEY")
        if primary_key:
            cfg.llm_providers.append(LLMProviderSpec(
                name=os.environ.get("RE_EMP_LLM_NAME", "primary"),
                base_url=os.environ.get("RE_EMP_LLM_BASE_URL",
                                       "https://api.openai.com/v1"),
                api_key=primary_key,
                model=os.environ.get("RE_EMP_LLM_MODEL", "gpt-4o-mini"),
                timeout=float(os.environ.get("RE_EMP_LLM_TIMEOUT", "60")),
                max_retries=int(os.environ.get("RE_EMP_LLM_MAX_RETRIES", "3"))))
        fb_key = os.environ.get("RE_EMP_LLM_FALLBACK_API_KEY")
        if fb_key:
            cfg.llm_providers.append(LLMProviderSpec(
                name="fallback",
                base_url=os.environ.get("RE_EMP_LLM_FALLBACK_BASE_URL",
                                       "https://api.openai.com/v1"),
                api_key=fb_key,
                model=os.environ.get("RE_EMP_LLM_FALLBACK_MODEL", "gpt-4o-mini")))
        return cfg


def build_llm(cfg: AppConfig, verbose: bool = True) -> ModelProvider:
    """有 key → ModelGateway（真实降级链）；否则 mock（打印告警）。"""
    if cfg.llm_providers:
        providers = [
            OpenAICompatibleProvider(p.name, p.base_url, p.api_key, p.model,
                                     p.timeout, p.max_retries)
            for p in cfg.llm_providers
        ]
        chain = " → ".join(f"{p.name}({p.model})" for p in providers)
        if verbose:
            print(f"[llm] model-gateway 降级链: {chain}")
        return ModelGateway(providers)
    if verbose:
        print("[llm] 未配置 API key（RE_EMP_LLM_API_KEY / OPENAI_API_KEY），"
              "降级为 MockResearchLLM（离线演示模式）")
    return MockResearchLLM()


def build_web(cfg: AppConfig, storage_root: str | Path,
              verbose: bool = True) -> RealWebBackend | MockWebBackend:
    """按 web_mode 与可用 key 选择后端；fetch_cache 落在 storage 下。"""
    cache_dir = cfg.fetch_cache_dir or Path(storage_root) / "fetch_cache"
    if cfg.web_mode in ("auto", "http") and (cfg.langsearch_key or cfg.brave_key
                                             or cfg.serp_key):
        if cfg.langsearch_key:
            from .tools.search_providers import LangSearchProvider
            search, src = LangSearchProvider(cfg.langsearch_key), "langsearch"
        elif cfg.brave_key:
            from .tools.search_providers import BraveSearchProvider
            search, src = BraveSearchProvider(cfg.brave_key), "brave"
        else:
            from .tools.search_providers import SerpApiProvider
            search, src = SerpApiProvider(cfg.serp_key), "serpapi"
        fetch = HttpxFetchBackend(cache_dir=cache_dir,
                                  ttl_seconds=cfg.fetch_ttl_seconds)
        if verbose:
            print(f"[web] 真实后端: search={src} fetch=httpx(磁盘缓存 {cache_dir})")
        return RealWebBackend(search, fetch)
    if cfg.web_mode in ("auto", "mcp") and cfg.you_mcp_enabled:
        from .tools.mcp_you import YouMCPWebBackend
        web = YouMCPWebBackend()
        if verbose:
            print("[web] You.com 免费 MCP（api.you.com/mcp?profile=free，无需 key，"
                  "IP 限流；RE_EMP_YOU_MCP=off 可禁用）")
        return web
    if verbose:
        if cfg.web_mode == "http":
            print("[web] RE_EMP_WEB=http 但缺少 LANGSEARCH_API_KEY/"
                  "BRAVE_API_KEY/SERPAPI_API_KEY，降级为 mock 搜索")
        else:
            print("[web] mock 搜索/抓取（离线演示模式；配置 LANGSEARCH_API_KEY/"
                  "BRAVE_API_KEY/SERPAPI_API_KEY 或保持默认启用 You.com MCP）")
    return MockWebBackend()
