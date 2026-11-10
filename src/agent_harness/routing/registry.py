"""Tool registration, allowlist filtering, and argument validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from agent_harness.core.errors import ToolError
from agent_harness.core.models import ToolDefinition
from agent_harness.safety.guardrails import validate_json_schema


@dataclass
class ToolContext:
    session_id: str
    task_id: str
    store: object


@dataclass
class ToolSpec:
    name: str
    description: str
    idempotent: bool
    source: str
    handler: Callable[[dict[str, object], ToolContext], str]
    args_model: type[BaseModel] | None = None
    schema: dict[str, object] | None = None

    def definition(self) -> ToolDefinition:
        if self.args_model is not None:
            parameters = self.args_model.model_json_schema()
        else:
            parameters = self.schema or {"type": "object", "properties": {}}
        return ToolDefinition(name=self.name, description=self.description, parameters=parameters)

    def validate(self, arguments: object) -> dict[str, object]:
        if self.args_model is not None:
            try:
                parsed = self.args_model.model_validate(arguments)
            except ValidationError as exc:
                fields = ", ".join(
                    ".".join(str(part) for part in error["loc"]) for error in exc.errors()
                )
                raise ToolError(
                    f"Invalid arguments for '{self.name}': check {fields or 'the payload'}"
                ) from exc
            return parsed.model_dump()
        return validate_json_schema(self.name, self.schema or {}, arguments)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, *, replace: bool = False) -> None:
        if spec.name in self._tools and not replace:
            raise ValueError(f"tool '{spec.name}' is already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def visible(self, allowed: set[str]) -> list[ToolDefinition]:
        return [spec.definition() for name, spec in sorted(self._tools.items()) if name in allowed]
