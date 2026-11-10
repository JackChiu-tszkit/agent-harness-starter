"""Local tools. Destructive memory deletion is not on the default allowlist."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from agent_harness.core.errors import ToolError
from agent_harness.tools.calculator import evaluate_expression


class CalculatorArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


class TextStatsArgs(BaseModel):
    text: str = Field(max_length=8000)


class StoreMemoryArgs(BaseModel):
    content: str = Field(min_length=1, max_length=2000)


class DeleteMemoryArgs(BaseModel):
    memory_id: int = Field(ge=1)


class EmptyArgs(BaseModel):
    pass


def current_utc_time() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def text_statistics(text: str) -> str:
    lines = text.splitlines()
    words = text.split()
    return f"characters={len(text)} words={len(words)} lines={len(lines)}"


def calculator(expression: str) -> str:
    return evaluate_expression(expression)


def store_memory(store: object, session_id: str, content: str) -> str:
    return store.add_memory(session_id, content)  # type: ignore[attr-defined]


def recall_memory(store: object, session_id: str) -> str:
    records = store.list_memories(session_id)  # type: ignore[attr-defined]
    if not records:
        return "no memories stored in this session"
    return "\n".join(f"{record.id}: {record.content}" for record in records)


def delete_memory(store: object, session_id: str, memory_id: int) -> str:
    deleted = store.delete_memory(session_id, memory_id)  # type: ignore[attr-defined]
    if not deleted:
        raise ToolError("memory was not found in this session")
    return f"deleted memory {memory_id}"
