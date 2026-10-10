"""事件溯源 + 检查点。

事件流是 append-only 的：任务的每一步（LLM 调用、工具调用、计划、检查点）
都写成不可变事件。崩溃恢复 = 加载 checkpoint + 重放 checkpoint 之后的事件。
M1 用 SQLite 单文件实现；生产切换 PostgreSQL 只改这一个类。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    stage: str
    subtask_cursor: int
    plan_version: int
    payload: dict = field(default_factory=dict)


class EventStore:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS event (
                task_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                ts REAL NOT NULL,
                type TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (task_id, seq))"""
        )
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS checkpoint (
                task_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                subtask_cursor INTEGER NOT NULL,
                plan_version INTEGER NOT NULL,
                payload TEXT NOT NULL)"""
        )
        self.conn.commit()

    # ---- 事件 ----
    def append(self, task_id: str, type_: str, payload: dict | None = None) -> int:
        with self.lock:
            seq = self.conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM event WHERE task_id = ?",
                (task_id,),
            ).fetchone()[0]
            self.conn.execute(
                "INSERT INTO event (task_id, seq, ts, type, payload) VALUES (?,?,?,?,?)",
                (
                    task_id,
                    seq,
                    time.time(),
                    type_,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
            self.conn.commit()
            return seq

    def replay(self, task_id: str, since_seq: int = 0) -> list[tuple[int, str, dict]]:
        rows = self.conn.execute(
            "SELECT seq, type, payload FROM event WHERE task_id = ? AND seq > ? ORDER BY seq",
            (task_id, since_seq),
        ).fetchall()
        return [(seq, t, json.loads(p)) for seq, t, p in rows]

    def latest_seq(self, task_id: str) -> int:
        return self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM event WHERE task_id = ?", (task_id,)
        ).fetchone()[0]

    # ---- 检查点 ----
    def save_checkpoint(
        self,
        task_id: str,
        stage: str,
        subtask_cursor: int,
        plan_version: int = 1,
        payload: dict | None = None,
    ) -> None:
        with self.lock:
            self.conn.execute(
                """INSERT INTO checkpoint (task_id, stage, subtask_cursor, plan_version, payload)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(task_id) DO UPDATE SET
                     stage = excluded.stage,
                     subtask_cursor = excluded.subtask_cursor,
                     plan_version = excluded.plan_version,
                     payload = excluded.payload""",
                (task_id, stage, subtask_cursor, plan_version,
                 json.dumps(payload or {}, ensure_ascii=False)),
            )
            self.conn.commit()
        self.append(
            task_id,
            "checkpoint.saved",
            {"stage": stage, "subtask_cursor": subtask_cursor, "plan_version": plan_version},
        )

    def load_checkpoint(self, task_id: str) -> Checkpoint | None:
        row = self.conn.execute(
            "SELECT stage, subtask_cursor, plan_version, payload FROM checkpoint WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            stage=row[0],
            subtask_cursor=row[1],
            plan_version=row[2],
            payload=json.loads(row[3]),
        )

    # ---- 恢复辅助 ----
    def scratchpad_for_subtask(self, task_id: str, subtask_index: int) -> list[str]:
        """从事件流重建某子任务的 scratchpad（崩溃恢复时不丢失循环内记忆）。"""
        lines: list[str] = []
        for _seq, type_, payload in self.replay(task_id):
            if type_ == "tool.called" and payload.get("subtask") == subtask_index and "scratch" in payload:
                lines.append(payload["scratch"])
        return lines

    def artifacts_written_in_subtask(
        self, task_id: str, subtask_index: int
    ) -> list[str]:
        ids: list[str] = []
        for _seq, type_, payload in self.replay(task_id):
            if (
                type_ == "tool.called"
                and payload.get("tool") == "write_artifact"
                and payload.get("subtask") == subtask_index
                and payload.get("ok")
            ):
                ids.append(payload["result"]["artifact_id"])
        return ids
