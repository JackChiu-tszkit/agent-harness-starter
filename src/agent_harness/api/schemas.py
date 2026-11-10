"""Request and response models for the local API."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SessionCreated(BaseModel):
    session_id: str
    status: str
    created_at: str


class MessageView(BaseModel):
    role: str
    content: str
    created_at: str


class SessionView(BaseModel):
    session_id: str
    status: str
    created_at: str
    updated_at: str
    messages: list[MessageView]
    memories: list[str]


class RunRequest(BaseModel):
    session_id: str = Field(pattern=_ID.pattern)
    message: str | None = Field(default=None, max_length=8000)
    resume_task_id: str | None = Field(default=None, pattern=_ID.pattern)
    approve: bool | None = None

    @model_validator(mode="after")
    def check_mode(self) -> RunRequest:
        if self.resume_task_id:
            if self.approve is None:
                raise ValueError("approve is required when resume_task_id is set")
            if self.message is not None:
                raise ValueError("message cannot be combined with resume_task_id")
        elif self.message is None or not self.message.strip():
            raise ValueError("message is required")
        return self


class RunResponse(BaseModel):
    task_id: str
    session_id: str
    status: str
    response: str | None
    error: str | None
    step_count: int
    correlation_id: str
    tool_calls: list[str]
    approval: dict[str, object] | None = None


class TaskView(BaseModel):
    task_id: str
    session_id: str
    status: str
    response: str | None
    error: str | None
    step_count: int
    correlation_id: str
    tool_calls: list[str]
    approval: dict[str, object] | None = None


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
