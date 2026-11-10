"""Deterministic mock model.

This is not an LLM. It pattern-matches a few demonstration requests and emits
tool calls. Tests and the three examples use it so they need no API key.
"""

from __future__ import annotations

import re
from decimal import Decimal

from agent_harness.core.models import ChatMessage, ModelRequest, ModelResponse, ToolCall
from agent_harness.safety.guardrails import looks_like_injection, visible_tool_text

_PRODUCTS = re.compile(r"(\d+)\s+products?\s+at\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
_SIMPLE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:multiplied by|times)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_DISCOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*discount", re.IGNORECASE)
_REMEMBER = re.compile(r"remember that\s+(.+)", re.IGNORECASE)
_STATS = re.compile(r"text statistics for:\s*(.*)", re.IGNORECASE | re.DOTALL)
_PRICE = re.compile(r"unit price is\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


def _percent_literal(raw: str) -> str:
    return format(Decimal(raw) / Decimal(100), "f")


def _substitute(value: object, previous: str) -> object:
    if not isinstance(value, str):
        return value
    if value == "{prev}":
        return previous
    return value.replace("{prev}", previous)


class MockModel:
    """Rule-based provider. Identified by name `mock` everywhere it is selected."""

    name = "mock"

    def complete(self, request: ModelRequest) -> ModelResponse:
        users = [message for message in request.messages if message.role == "user"]
        if not users:
            return ModelResponse(content="No user request was provided.", finish_reason="stop")
        latest = users[-1].content
        tool_messages = _tools_after_latest_user(request.messages)
        if _REMEMBER.search(latest):
            return _remember(latest, tool_messages)
        if "unit price" in latest.lower() and "remember" in latest.lower():
            return _recall_price(request.memories)
        if tool_messages:
            payload = visible_tool_text(tool_messages[-1].content)
            if payload.startswith("Tool error:"):
                return ModelResponse(content=f"The tool failed: {payload}", finish_reason="stop")
        operations = _operations(latest)
        if operations:
            return _next_operation(latest, operations, tool_messages)
        if looks_like_injection(latest):
            return ModelResponse(
                content="I will not follow instructions that try to override the harness policy.",
                finish_reason="stop",
            )
        return ModelResponse(
            content=(
                "The mock model can calculate, report the current UTC time, "
                "summarize text, and recall facts stored in this session."
            ),
            finish_reason="stop",
        )


def _tools_after_latest_user(messages: list[ChatMessage]) -> list[ChatMessage]:
    last_user = max(index for index, message in enumerate(messages) if message.role == "user")
    return [message for message in messages[last_user + 1 :] if message.role == "tool"]


def _remember(latest: str, tool_messages: list[ChatMessage]) -> ModelResponse:
    if not tool_messages:
        fact = _REMEMBER.search(latest)
        content = fact.group(1).strip().rstrip(".") if fact else latest.strip()
        return ModelResponse(
            tool_calls=[ToolCall(name="store_memory", arguments={"content": content})],
            finish_reason="tool_calls",
        )
    stored = visible_tool_text(tool_messages[-1].content)
    return ModelResponse(content=f"Stored for this session: {stored}.", finish_reason="stop")


def _recall_price(memories: list[str]) -> ModelResponse:
    for memory in memories:
        match = _PRICE.search(memory)
        if match:
            return ModelResponse(
                content=f"The remembered unit price is {match.group(1)}.",
                finish_reason="stop",
            )
    return ModelResponse(
        content="I do not have a remembered unit price in this session.",
        finish_reason="stop",
    )


def _operations(text: str) -> list[tuple[str, dict[str, object]]]:
    discount = _DISCOUNT.search(text)
    products = _PRODUCTS.search(text)
    simple = _SIMPLE.search(text)
    if products and discount:
        percent = _percent_literal(discount.group(1))
        return [
            ("calculator", {"expression": f"{products.group(1)} * {products.group(2)}"}),
            ("calculator", {"expression": "{prev} * (1 - " + percent + ")"}),
        ]
    if simple and discount is None:
        return [
            ("calculator", {"expression": f"{simple.group(1)} * {simple.group(2)}"}),
        ]
    lowered = text.lower()
    if "utc" in lowered or "current time" in lowered:
        return [("current_utc_time", {})]
    stats = _STATS.search(text)
    if stats:
        return [("text_statistics", {"text": stats.group(1).strip()})]
    return []


def _next_operation(
    latest: str,
    operations: list[tuple[str, dict[str, object]]],
    tool_messages: list[ChatMessage],
) -> ModelResponse:
    if len(tool_messages) < len(operations):
        name, arguments = operations[len(tool_messages)]
        if tool_messages:
            previous = visible_tool_text(tool_messages[-1].content)
            arguments = {key: _substitute(value, previous) for key, value in arguments.items()}
        return ModelResponse(
            tool_calls=[ToolCall(name=name, arguments=arguments)],
            finish_reason="tool_calls",
        )
    result = visible_tool_text(tool_messages[-1].content)
    return ModelResponse(content=_final_text(latest, result), finish_reason="stop")


def _final_text(latest: str, result: str) -> str:
    if _DISCOUNT.search(latest):
        return f"The discounted total is {result}."
    simple = _SIMPLE.search(latest)
    if simple:
        return f"{simple.group(1)} multiplied by {simple.group(2)} is {result}."
    if "utc" in latest.lower() or "current time" in latest.lower():
        return f"The current UTC time is {result}."
    if _STATS.search(latest):
        return f"Text statistics: {result}."
    return result
