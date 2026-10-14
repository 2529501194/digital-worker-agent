"""工具路由器：统一工具协议 + 岗位白名单过滤。

每个工具调用：参数校验 → 执行 → 结果截断 → 返回 ToolResult（含 scratch 行）。
工具失败不终止循环：错误作为 observation 返回给 LLM，由其决定换路/跳过。
"""
from __future__ import annotations

from dataclasses import dataclass

from .artifact_store import ArtifactStore
from .web import MockWebBackend, ToolResult

MAX_RESULT_CHARS = 2000  # 防单次工具结果灌爆上下文

# 研究员岗位白名单（角色配置；CLI 与 Web 服务共用）
RESEARCHER_WHITELIST = ["web_search", "web_fetch", "write_artifact",
                       "read_artifact", "list_artifacts"]


def _truncate(s: str, n: int = MAX_RESULT_CHARS) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[截断，共 {len(s)} 字符]"


class ToolRouter:
    def __init__(self, web: MockWebBackend, artifacts: ArtifactStore,
                 whitelist: list[str] | None = None):
        self.web = web
        self.artifacts = artifacts
        self.whitelist = set(whitelist or [
            "web_search", "web_fetch", "write_artifact", "read_artifact", "list_artifacts"])
        self._handlers = {
            "web_search": self._web_search,
            "web_fetch": self._web_fetch,
            "write_artifact": self._write_artifact,
            "read_artifact": self._read_artifact,
            "list_artifacts": self._list_artifacts,
        }

    # ---- 对外入口 ----
    def run(self, task_id: str, action_name: str, args: dict, subtask: int | None = None) -> ToolResult:
        if action_name not in self.whitelist:
            return ToolResult(ok=False, error=f"工具 {action_name} 不在岗位白名单内")
        handler = self._handlers.get(action_name)
        if handler is None:
            return ToolResult(ok=False, error=f"工具 {action_name} 未注册")
        return handler(task_id, args, subtask)

    # ---- 工具实现 ----
    def _web_search(self, task_id: str, args: dict, subtask: int | None) -> ToolResult:
        # 参数别名：模型常把搜索词写成 q/keyword（与 read_artifact 的别名同一思路）
        query = str(args.get("query") or args.get("q") or args.get("keyword") or "").strip()
        if not query:
            return ToolResult(
                ok=False, error="web_search 缺少 query 参数",
                scratch=('web_search => ERROR 缺少 query 参数。'
                         '正确写法：{"action":"web_search","args":{"query":"<具体搜索词>"}}，'
                         '参数必须放在 args 对象内，不要把 query 写在顶层或留空'))
        try:
            results = self.web.search(query, limit=int(args.get("limit", 3)))
        except Exception as e:  # 真实搜索 API 失败 → 错误交回 LLM 决策，不终止循环
            return ToolResult(ok=False, error=f"web_search 失败: {e}",
                              scratch=f'web_search "{query}" => ERROR {e}')
        if not results:
            line = f'web_search "{query}" => 0 results'
        else:
            items = "; ".join(f"[{i+1}] {r['url']} :: {r['title']}"
                              for i, r in enumerate(results))
            line = f'web_search "{query}" => {len(results)} results: {items}'
        return ToolResult(ok=True,
                          data={"query": query, "results": results},
                          scratch=line)

    def _web_fetch(self, task_id: str, args: dict, subtask: int | None) -> ToolResult:
        url = str(args.get("url") or args.get("link") or args.get("href") or "").strip()
        if not url:
            return ToolResult(
                ok=False, error="web_fetch 缺少 url 参数",
                scratch=('web_fetch => ERROR 缺少 url 参数。'
                         '正确写法：{"action":"web_fetch","args":{"url":"<完整 URL>"}}，'
                         'URL 取自 web_search 结果或证据工件 refs'))
        content, err = self.web.fetch(url)
        if err:
            return ToolResult(ok=False, error=err,
                              scratch=f"web_fetch {url} => ERROR {err}")
        snippet = content[:150].replace("\n", " ")
        return ToolResult(ok=True,
                          data={"url": url, "markdown": _truncate(content)},
                          scratch=f"web_fetch {url} => OK | {snippet}")

    def _write_artifact(self, task_id: str, args: dict, subtask: int | None) -> ToolResult:
        type_ = args.get("type", "note")
        title = (args.get("title") or "untitled").strip()
        content = args.get("content") or ""
        refs = args.get("refs") or []
        if not content.strip():
            return ToolResult(ok=False, error="write_artifact 内容为空",
                              scratch=f"write_artifact [{type_}] {title} => ERROR 内容为空")
        art = self.artifacts.write(task_id, type_, title, _truncate(content, 20000), refs)
        return ToolResult(ok=True,
                          data={"artifact_id": art.id},
                          scratch=f"write_artifact [{type_}] {title} => {art.id}")

    def _read_artifact(self, task_id: str, args: dict, subtask: int | None) -> ToolResult:
        # 兼容常见别名（模型没看到参数表时的常见误传：path/artifact_id）
        art_id = str(args.get("id") or args.get("artifact_id")
                     or args.get("path") or "").strip()
        if not art_id:
            return ToolResult(
                ok=False, error="read_artifact 缺少 id 参数",
                scratch=(f"read_artifact => ERROR 缺少 id 参数（args 必须含 "
                         f'"id"，可用 list_artifacts 查看当前工件 id）'))
        try:
            art = self.artifacts.read(task_id, art_id)
        except KeyError as e:
            items = self.artifacts.list(task_id)
            hint = "; ".join(f"{m['id']}[{m['type']}]{m['title']}"
                             for m in items[:5]) or "（当前无工件）"
            return ToolResult(ok=False, error=str(e),
                              scratch=(f"read_artifact {art_id} => ERROR {e}；"
                                       f"当前工件: {hint}"))
        return ToolResult(ok=True,
                          data={"artifact": art.__dict__},
                          scratch=f"read_artifact {art_id} => OK")

    def _list_artifacts(self, task_id: str, args: dict, subtask: int | None) -> ToolResult:
        items = self.artifacts.list(task_id, args.get("type"))
        line = "list_artifacts => " + "; ".join(f"{m['id']}[{m['type']}]{m['title']}" for m in items)
        return ToolResult(ok=True, data={"items": items}, scratch=line)
