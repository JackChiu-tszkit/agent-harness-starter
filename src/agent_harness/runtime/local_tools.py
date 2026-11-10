"""Register the in-process tools that need harness state."""

from __future__ import annotations

from agent_harness.core.errors import ToolError
from agent_harness.routing.registry import ToolContext, ToolRegistry, ToolSpec
from agent_harness.tools.builtin import (
    CalculatorArgs,
    DeleteMemoryArgs,
    EmptyArgs,
    StoreMemoryArgs,
    TextStatsArgs,
    calculator,
    current_utc_time,
    delete_memory,
    recall_memory,
    store_memory,
    text_statistics,
)


def build_local_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="Evaluate a basic arithmetic expression using only numbers and + - * /.",
            idempotent=True,
            source="local",
            args_model=CalculatorArgs,
            handler=lambda arguments, _context: calculator(str(arguments["expression"])),
        )
    )
    registry.register(
        ToolSpec(
            name="current_utc_time",
            description="Return the current UTC time in ISO 8601 format.",
            idempotent=False,
            source="local",
            args_model=EmptyArgs,
            handler=lambda _arguments, _context: current_utc_time(),
        )
    )
    registry.register(
        ToolSpec(
            name="text_statistics",
            description="Count characters, words, and lines in a text value.",
            idempotent=True,
            source="local",
            args_model=TextStatsArgs,
            handler=lambda arguments, _context: text_statistics(str(arguments["text"])),
        )
    )
    registry.register(
        ToolSpec(
            name="store_memory",
            description="Store a fact in this session's long-term memory.",
            idempotent=False,
            source="local",
            args_model=StoreMemoryArgs,
            handler=_store,
        )
    )
    registry.register(
        ToolSpec(
            name="recall_memory",
            description="List facts stored in this session.",
            idempotent=True,
            source="local",
            args_model=EmptyArgs,
            handler=_recall,
        )
    )
    registry.register(
        ToolSpec(
            name="delete_memory",
            description="Delete one memory row that belongs to this session.",
            idempotent=False,
            source="local",
            args_model=DeleteMemoryArgs,
            handler=_delete,
        )
    )
    return registry


def _store(arguments: dict[str, object], context: ToolContext) -> str:
    return store_memory(context.store, context.session_id, str(arguments["content"]))


def _recall(_arguments: dict[str, object], context: ToolContext) -> str:
    return recall_memory(context.store, context.session_id)


def _delete(arguments: dict[str, object], context: ToolContext) -> str:
    memory_id = arguments["memory_id"]
    if isinstance(memory_id, bool) or not isinstance(memory_id, int):
        raise ToolError("Invalid arguments for 'delete_memory': memory_id must be an integer")
    return delete_memory(context.store, context.session_id, memory_id)
