class C:
        def _read_artifact(self, task_id: str, args: dict, subtask: int | None) -> ToolResult:
            # 兼容常见别名（模型没看到参数表时的常见误传：path/artifact_id）
            art_id = str(args.get("id") or args.get("artifact_id")
                         or args.get("path") or "").strip()
            if not art_id:
                return ToolResult(
                    ok=False, error="read_artifact 缺少 id 参数",
                    scratch=(f"read_artifact => ERROR 缺少 id 参数（args 必须含 "
                             f'"id"，可用 list_artifacts 查看当前工件 id）"))
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
