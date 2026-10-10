"""搜索与网页抓取工具后端。

- MockWebBackend（M1）：确定性内置语料，离线可测；
- RealWebBackend（M2）：真实 SearchProvider（Brave/SerpApi）+ HttpxFetchBackend
  （httpx 抓取 + HTML→Markdown + 磁盘缓存）。
两者接口一致：search(query, limit) / fetch(url) / fetch_counts。
fetch 带缓存与调用计数 —— 崩溃恢复重放时不重复抓同一 URL。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .fetch_backends import HttpxFetchBackend
from .search_providers import MockSearchProvider, SearchProvider


PAGES: dict[str, dict[str, str]] = {
    "https://docs.claude-code.dev/arch": {
        "title": "Claude Code Architecture",
        "content": (
            "# Claude Code Architecture\n"
            "Claude Code 是终端内的 agentic coding 工具，核心是 ReAct 式 agent 循环：\n"
            "模型先规划（读取任务、拆解目标），再调用工具（bash、read、edit、write、grep），\n"
            "观察结果后迭代。任务规划依赖系统提示词 + 用户目标，无显式 DAG。\n"
            "上下文管理采用滑动窗口 + 自动压缩（compact），接近上限时由模型生成会话摘要。\n"
            "工具使用受权限模型约束：危险命令需用户确认。"
        ),
    },
    "https://openai.com/codex-cli": {
        "title": "OpenAI Codex CLI",
        "content": (
            "# OpenAI Codex CLI\n"
            "Codex CLI 是本地运行的编码 agent，模型（codex 系列）在沙箱容器内执行 shell 与文件操作。\n"
            "规划方式：模型内部规划为主，配合 AGENTS.md 提供项目级指令。\n"
            "工具面较窄：shell、apply_patch、文件读取；强调沙箱隔离（容器 + 网络白名单）。\n"
            "上下文管理：上下文窗口内滚动，超限时自动总结历史。"
        ),
    },
    "https://cursor.com/docs/agent": {
        "title": "Cursor Agent",
        "content": (
            "# Cursor Agent\n"
            "Cursor 的 Agent 模式运行在 IDE 内：模型规划多文件编辑计划，调用工具（终端命令、\n"
            "文件编辑、代码库语义检索）逐步执行，可自主多轮迭代直至完成。\n"
            "上下文管理亮点：codebase 索引 + 语义检索（embedding 检索相关代码）注入上下文，\n"
            "并支持 @docs 等外部知识源。工具使用有 allowlist 与自动批准开关。"
        ),
    },
    "https://docs.all-hands.dev/architecture": {
        "title": "OpenHands Architecture",
        "content": (
            "# OpenHands (OpenDevin) Architecture\n"
            "OpenHands 采用显式事件流架构：agent 动作（thought / action / observation）全部作为\n"
            "事件进入 EventStream，由控制器驱动，因此可暂停、可恢复、可回放。\n"
            "工具使用：CodeActAgent 通过执行 Python/shell 完成多步任务，规划为模型内隐规划 + 反思。\n"
            "上下文管理：事件流压缩（summarization）与记忆模块（micro/long-term memory）。"
        ),
    },
    "https://github.com/features/copilot": {
        "title": "GitHub Copilot & Copilot coding agent",
        "content": (
            "# GitHub Copilot\n"
            "GitHub Copilot 从行内补全演进到 agentic 编码：Copilot coding agent 可在云端接收\n"
            "issue 任务，自主规划多步修改、调用 shell 与文件工具、运行测试并提交 PR。\n"
            "规划：模型内隐规划 + 任务拆解为子步骤；工具：shell、文件编辑、代码搜索；\n"
            "上下文管理：仓库索引 + 相关代码检索注入，会话历史滚动压缩。"
        ),
    },
    "https://aider.chat/docs/": {
        "title": "Aider: pair programming in your terminal",
        "content": (
            "# Aider Architecture\n"
            "Aider 是开源终端编码 agent，git 原生：每次修改自动提交，可回滚。\n"
            "核心循环：用户指令 → 模型规划 → 以 edit format（diff / whole / unified）输出代码变更\n"
            "→ 应用并运行测试 → 反馈给模型迭代（ReAct 式）。\n"
            "上下文管理亮点：repo map（基于 tree-sitter 的符号排名）把全仓库结构压缩进上下文窗口。"
        ),
    },
    "https://docs.devin.ai/overview": {
        "title": "Devin: AI Software Engineer",
        "content": (
            "# Devin (Cognition)\n"
            "Devin 是面向长时程任务的自主软件工程 agent：接收目标后自行拆解计划，\n"
            "在云端沙箱内使用自己的 shell、编辑器与浏览器多轮执行，可持续数小时。\n"
            "规划：显式计划 + 执行中重规划；工具：shell / editor / browser / 代码库搜索；\n"
            "上下文管理：内部记忆与 scratchpad 机制，超出窗口时总结旧轨迹。"
        ),
    },
}


class MockWebBackend:
    """确定性搜索/抓取后端（M1）。委托 MockSearchProvider + 内存缓存。"""

    name = "mock"

    def __init__(self, pages: dict[str, dict[str, str]] | None = None):
        self.pages = pages if pages is not None else PAGES
        self._search = MockSearchProvider(self.pages)
        self.fetch_cache: dict[str, str] = {}
        self.fetch_counts: dict[str, int] = {}

    def search(self, query: str, limit: int = 3) -> list[dict]:
        return self._search.search(query, limit)

    def fetch(self, url: str) -> tuple[str | None, str | None]:
        """返回 (markdown, error)。缓存命中不增加真实抓取语义，但计数 +1 以便审计。"""
        self.fetch_counts[url] = self.fetch_counts.get(url, 0) + 1
        if url in self.fetch_cache:
            return self.fetch_cache[url], None
        page = self.pages.get(url)
        if page is None:
            return None, f"404: 无法抓取 {url}"
        self.fetch_cache[url] = page["content"]
        return page["content"], None


class RealWebBackend:
    """M2 真实后端：真实搜索 Provider + httpx 抓取（带磁盘缓存）。"""

    def __init__(self, search: SearchProvider, fetch: HttpxFetchBackend):
        self.search_provider = search
        self.fetch_backend = fetch
        self.fetch_counts = fetch.fetch_counts

    @property
    def name(self) -> str:
        return f"{self.search_provider.name}+http"

    def search(self, query: str, limit: int = 3) -> list[dict]:
        return self.search_provider.search(query, limit)

    def fetch(self, url: str) -> tuple[str | None, str | None]:
        return self.fetch_backend.fetch(url)


@dataclass
class ToolResult:
    ok: bool
    data: dict = field(default_factory=dict)
    error: str | None = None
    scratch: str = ""   # 写回 LLM 上下文的一行摘要
