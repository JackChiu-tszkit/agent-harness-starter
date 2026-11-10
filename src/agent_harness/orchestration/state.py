"""State carried by the single LangGraph workflow."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    session_id: str
    task_id: str
    correlation_id: str
    user_message: str
    system_prompt: str
    memories: list[str]
    history: list[dict[str, str]]
    transcript: list[dict[str, str | None]]
    pending_tool: dict | None
    tool_output: str | None
    tool_error: str | None
    step_count: int
    max_steps: int
    status: str
    response: str | None
    error: str | None
    tool_calls: list[str]
