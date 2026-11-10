"""Input, output, argument, and tool-output checks.

These checks are deterministic boundaries. They do not make prompt text a
security control by itself.
"""

from __future__ import annotations

import re

from agent_harness.core.errors import InputError, OutputError, ToolError

_INJECTION = re.compile(
    r"(ignore\s+(all|any|previous)\s+instructions|"
    r"reveal\s+the\s+system\s+prompt|"
    r"you\s+are\s+now\b)",
    re.IGNORECASE,
)
_WRAPPER = re.compile(
    r"<untrusted_tool_output(?:\s+[^>]*)?>\s*(.*?)\s*</untrusted_tool_output>",
    re.DOTALL,
)
_FLAG = "[flagged possible prompt injection] "
_TYPE_CHECKS = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def validate_input(text: str, max_chars: int) -> str:
    if text is None or not str(text).strip():
        raise InputError("message is empty")
    cleaned = str(text).strip()
    if len(cleaned) > max_chars:
        raise InputError(f"message exceeds the {max_chars} character limit")
    return cleaned


def validate_output(text: str | None, max_chars: int) -> str:
    if text is None or not str(text).strip():
        raise OutputError("model returned an empty response")
    cleaned = str(text).strip()
    if len(cleaned) > max_chars:
        raise OutputError(f"response exceeds the {max_chars} character limit")
    return cleaned


def looks_like_injection(text: str) -> bool:
    return _INJECTION.search(text) is not None


def wrap_tool_output(tool_name: str, content: str) -> str:
    """Keep tool output in a delimiter the system prompt tells the model not to obey."""

    body = content.strip()
    if looks_like_injection(body):
        body = f"{_FLAG}{body}"
    return f'<untrusted_tool_output name="{tool_name}">\n{body}\n</untrusted_tool_output>'


def visible_tool_text(content: str) -> str:
    match = _WRAPPER.search(content)
    text = match.group(1).strip() if match else content.strip()
    if text.startswith(_FLAG):
        return text[len(_FLAG) :].strip()
    return text


def validate_json_schema(
    tool_name: str,
    schema: dict[str, object],
    arguments: object,
) -> dict[str, object]:
    if not isinstance(arguments, dict):
        raise ToolError(f"Invalid arguments for '{tool_name}': arguments must be an object")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ToolError(f"Invalid arguments for '{tool_name}': tool schema is unusable")
    missing = [name for name in required if name not in arguments]
    if missing:
        joined = ", ".join(map(str, missing))
        raise ToolError(f"Invalid arguments for '{tool_name}': missing {joined}")
    unexpected = [name for name in arguments if name not in properties]
    if unexpected:
        raise ToolError(
            f"Invalid arguments for '{tool_name}': unexpected {', '.join(map(str, unexpected))}"
        )
    cleaned: dict[str, object] = {}
    for name, value in arguments.items():
        spec = properties.get(name, {})
        expected = spec.get("type") if isinstance(spec, dict) else None
        if expected == "integer" and isinstance(value, bool):
            raise ToolError(f"Invalid arguments for '{tool_name}': {name} must be an integer")
        if expected in _TYPE_CHECKS and not isinstance(value, _TYPE_CHECKS[expected]):
            raise ToolError(f"Invalid arguments for '{tool_name}': {name} has the wrong type")
        if isinstance(value, str) and len(value) > 8000:
            raise ToolError(f"Invalid arguments for '{tool_name}': {name} is too long")
        cleaned[str(name)] = value
    return cleaned
