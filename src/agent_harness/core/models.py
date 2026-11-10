"""Domain records and the model-provider contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_name: str | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, object]


class ModelRequest(BaseModel):
    """What the router shows a model. The system prompt is trusted; messages are not."""

    system_prompt: str
    messages: list[ChatMessage]
    memories: list[str] = Field(default_factory=list)
    tools: list[ToolDefinition] = Field(default_factory=list)


class ModelResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: Literal["stop", "tool_calls", "error"] = "stop"


class ModelProvider(Protocol):
    """Provider-specific clients implement this and nothing else in the graph."""

    name: str

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a final message or one tool call. Must not execute tools."""


@dataclass
class SessionRecord:
    id: str
    status: str
    created_at: str
    updated_at: str


@dataclass
class MessageRecord:
    role: str
    content: str
    created_at: str


@dataclass
class MemoryRecord:
    id: int
    content: str
    created_at: str


@dataclass
class TaskRecord:
    id: str
    session_id: str
    status: str
    request: str
    response: str | None
    error: str | None
    step_count: int
    correlation_id: str
    tool_calls: list[str]
    approval: dict[str, object] | None
    created_at: str
    updated_at: str


@dataclass
class RunResult:
    task_id: str
    session_id: str
    status: str
    response: str | None
    error: str | None
    step_count: int
    correlation_id: str
    tool_calls: list[str] = field(default_factory=list)
    approval: dict[str, object] | None = None


@dataclass
class SessionContext:
    session_id: str
    history: list[dict[str, str]]
    memories: list[str]
