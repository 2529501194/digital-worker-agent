    def _read_artifact(self, task_id: str, args: dict, subtask: int | None) -> ToolResult:
        # 兼容常见别名（模型没看到参数表时的常见误传：path/artifact_id）
        art_id = str(args.get("id") or args.get("artifact_id")
                     or args.get("path") or "").strip()
        if not art_id:
            return ToolResult(
                ok=False, error="read_artifact 缺少 id 参数",
                scratch=(f"read_artifact => ERROR 缺少 id 参数（args 必须含 "
                         f'"id"，可用 list_artifacts 查看当前工件 id）"))