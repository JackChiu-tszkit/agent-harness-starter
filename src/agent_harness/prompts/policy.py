"""Execution policy. Privileged tools stay deny-by-default unless allowlisted."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_harness.config import Settings


class ExecutionPolicy(BaseModel):
    max_steps: int
    tool_timeout_seconds: float
    max_retries: int
    max_input_chars: int
    max_context_chars: int
    max_output_chars: int
    allowed_tools: set[str] = Field(default_factory=set)
    approval_tools: set[str] = Field(default_factory=set)
    prompt_version: str = "v1"


def policy_from_settings(settings: Settings) -> ExecutionPolicy:
    return ExecutionPolicy(
        max_steps=settings.max_steps,
        tool_timeout_seconds=settings.tool_timeout_seconds,
        max_retries=settings.max_retries,
        max_input_chars=settings.max_input_chars,
        max_context_chars=settings.max_context_chars,
        max_output_chars=settings.max_output_chars,
        allowed_tools=set(settings.allowed_tools),
        approval_tools=set(settings.approval_tools),
    )
