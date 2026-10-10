"""工件库：研究任务的中间成果（笔记/证据/大纲/草稿/计划）。

M1：本地文件系统（artifacts/{task_id}/{id}.md + meta.json）。
生产：对象存储 + 元数据入库，接口不变。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Artifact:
    id: str
    type: str          # plan | evidence | note | outline | draft | report
    title: str
    content: str
    refs: list[str] = field(default_factory=list)
    version: int = 1


_TYPE_PREFIX = {"plan": "P", "evidence": "E", "note": "N",
                "outline": "O", "draft": "D", "report": "R"}


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _task_dir(self, task_id: str) -> Path:
        d = self.root / task_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _meta_path(self, task_id: str) -> Path:
        return self._task_dir(task_id) / "meta.json"

    def _load_meta(self, task_id: str) -> dict[str, dict[str, Any]]:
        p = self._meta_path(task_id)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def _save_meta(self, task_id: str, meta: dict[str, dict[str, Any]]) -> None:
        self._meta_path(task_id).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def write(self, task_id: str, type_: str, title: str,
              content: str, refs: list[str] | None = None) -> Artifact:
        prefix = _TYPE_PREFIX.get(type_, "A")
        meta = self._load_meta(task_id)
        n = sum(1 for m in meta.values() if m["type"] == type_) + 1
        art_id = f"{prefix}-{n}"
        (self._task_dir(task_id) / f"{art_id}.md").write_text(content, encoding="utf-8")
        meta[art_id] = {"id": art_id, "type": type_, "title": title,
                        "refs": refs or [], "version": 1}
        self._save_meta(task_id, meta)
        return Artifact(id=art_id, type=type_, title=title,
                        content=content, refs=refs or [])

    def read(self, task_id: str, artifact_id: str) -> Artifact:
        meta = self._load_meta(task_id)
        if artifact_id not in meta:
            raise KeyError(f"artifact 不存在: {artifact_id}")
        m = meta[artifact_id]
        content = (self._task_dir(task_id) / f"{artifact_id}.md").read_text(encoding="utf-8")
        return Artifact(id=artifact_id, type=m["type"], title=m["title"],
                        content=content, refs=m.get("refs", []),
                        version=m.get("version", 1))

    def list(self, task_id: str, type_: str | None = None) -> list[dict]:
        out = []
        for m in self._load_meta(task_id).values():
            if type_ is None or m["type"] == type_:
                out.append(m)
        return out

    def full(self, task_id: str, type_: str | None = None) -> list[dict]:
        """list + 内容（供 WRITE 阶段组装上下文用；内容截断在调用方处理）。"""
        return [self.read(task_id, m["id"]).__dict__ for m in self.list(task_id, type_)]
