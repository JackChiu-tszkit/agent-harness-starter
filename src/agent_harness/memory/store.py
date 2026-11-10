"""SQLite persistence for sessions, conversation history, tasks, and memories.

Workflow checkpoints live in LangGraph's SQLite saver, not in these tables.
Conversation history is durable session state. Memory rows are explicit
long-term facts. Task rows are the status of one run.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from agent_harness.core.errors import HarnessError
from agent_harness.core.models import MemoryRecord, MessageRecord, SessionRecord, TaskRecord


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StateStore:
    def __init__(self, path: str) -> None:
        database = Path(path)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(database)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._setup()

    def _setup(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request TEXT NOT NULL,
                    response TEXT,
                    error TEXT,
                    step_count INTEGER NOT NULL DEFAULT 0,
                    correlation_id TEXT NOT NULL,
                    tool_calls_json TEXT NOT NULL DEFAULT '[]',
                    approval_json TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create_session(self) -> SessionRecord:
        session_id = str(uuid.uuid4())
        timestamp = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, status, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, "active", timestamp, timestamp),
            )
            self._conn.commit()
        return SessionRecord(session_id, "active", timestamp, timestamp)

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SessionRecord(row["id"], row["status"], row["created_at"], row["updated_at"])

    def require_active_session(self, session_id: str) -> SessionRecord:
        session = self.get_session(session_id)
        if session is None:
            raise HarnessError("session_not_found", "session was not found", 404)
        if session.status != "active":
            raise HarnessError("session_closed", "session is closed", 409)
        return session

    def close_session(self, session_id: str) -> SessionRecord:
        session = self.get_session(session_id)
        if session is None:
            raise HarnessError("session_not_found", "session was not found", 404)
        timestamp = _now()
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                ("closed", timestamp, session_id),
            )
            self._conn.commit()
        return SessionRecord(session_id, "closed", session.created_at, timestamp)

    def delete_session(self, session_id: str) -> list[str]:
        """Delete one session and return its task ids so checkpoints can be removed."""

        if self.get_session(session_id) is None:
            raise HarnessError("session_not_found", "session was not found", 404)
        with self._lock:
            task_ids = [
                row["id"]
                for row in self._conn.execute(
                    "SELECT id FROM tasks WHERE session_id = ?", (session_id,)
                ).fetchall()
            ]
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM tasks WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()
        return task_ids

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, _now()),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id)
            )
            self._conn.commit()

    def list_messages(self, session_id: str) -> list[MessageRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [MessageRecord(row["role"], row["content"], row["created_at"]) for row in rows]

    def add_memory(self, session_id: str, content: str) -> str:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (session_id, content, created_at) VALUES (?, ?, ?)",
                (session_id, content, _now()),
            )
            self._conn.commit()
        return content

    def list_memories(self, session_id: str) -> list[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content, created_at FROM memories WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [MemoryRecord(row["id"], row["content"], row["created_at"]) for row in rows]

    def delete_memory(self, session_id: str, memory_id: int) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE id = ? AND session_id = ?",
                (memory_id, session_id),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def create_task(self, session_id: str, request: str, correlation_id: str) -> TaskRecord:
        task_id = str(uuid.uuid4())
        timestamp = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tasks (
                    id, session_id, status, request, step_count, correlation_id,
                    tool_calls_json, cancel_requested, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    session_id,
                    "running",
                    request,
                    0,
                    correlation_id,
                    "[]",
                    0,
                    timestamp,
                    timestamp,
                ),
            )
            self._conn.commit()
        return self.get_task(task_id)  # type: ignore[return-value]

    def update_task(
        self,
        task_id: str,
        *,
        status: str,
        response: str | None = None,
        error: str | None = None,
        step_count: int = 0,
        tool_calls: list[str] | None = None,
        approval: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE tasks
                SET status = ?, response = ?, error = ?, step_count = ?,
                    tool_calls_json = ?, approval_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    response,
                    error,
                    step_count,
                    json.dumps(tool_calls or []),
                    json.dumps(approval) if approval is not None else None,
                    _now(),
                    task_id,
                ),
            )
            self._conn.commit()

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        approval = json.loads(row["approval_json"]) if row["approval_json"] else None
        return TaskRecord(
            id=row["id"],
            session_id=row["session_id"],
            status=row["status"],
            request=row["request"],
            response=row["response"],
            error=row["error"],
            step_count=row["step_count"],
            correlation_id=row["correlation_id"],
            tool_calls=json.loads(row["tool_calls_json"]),
            approval=approval,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def require_task(self, task_id: str, session_id: str | None = None) -> TaskRecord:
        task = self.get_task(task_id)
        if task is None or (session_id is not None and task.session_id != session_id):
            raise HarnessError("task_not_found", "task was not found", 404)
        return task

    def request_cancel(self, task_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                (_now(), task_id),
            )
            self._conn.commit()

    def is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT cancel_requested FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])
