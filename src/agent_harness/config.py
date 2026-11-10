"""Process configuration. API keys come only from the environment."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ModelProviderName = Literal["mock", "openai"]
OtelExporterName = Literal["console", "none", "memory"]

DEFAULT_ALLOWED_TOOLS = (
    "calculator",
    "current_utc_time",
    "text_statistics",
    "store_memory",
    "recall_memory",
)
DEFAULT_APPROVAL_TOOLS = ("delete_memory",)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Runtime settings. The mock provider is the default and needs no API key."""

    model_provider: ModelProviderName = "mock"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 30.0
    database_path: str = "./data/harness.db"
    max_steps: int = 8
    tool_timeout_seconds: float = 5.0
    max_retries: int = 2
    max_input_chars: int = 8000
    max_context_chars: int = 12000
    max_output_chars: int = 8000
    log_level: str = "INFO"
    otel_exporter: OtelExporterName = "console"
    enable_mcp: bool = True
    allowed_tools: list[str] = Field(default_factory=lambda: list(DEFAULT_ALLOWED_TOOLS))
    approval_tools: list[str] = Field(default_factory=lambda: list(DEFAULT_APPROVAL_TOOLS))
    host: str = "127.0.0.1"
    port: int = 8000

    @field_validator("max_steps")
    @classmethod
    def check_steps(cls, value: int) -> int:
        if not 1 <= value <= 50:
            raise ValueError("max_steps must be between 1 and 50")
        return value

    @field_validator("max_retries")
    @classmethod
    def check_retries(cls, value: int) -> int:
        if not 0 <= value <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        return value

    @field_validator("tool_timeout_seconds", "openai_timeout_seconds")
    @classmethod
    def check_timeout(cls, value: float) -> float:
        if not 0 < value <= 120:
            raise ValueError("timeouts must be greater than 0 and at most 120 seconds")
        return value

    @field_validator("max_input_chars", "max_context_chars", "max_output_chars")
    @classmethod
    def check_chars(cls, value: int) -> int:
        if value < 1:
            raise ValueError("character limits must be positive")
        return value

    @field_validator("log_level")
    @classmethod
    def check_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalized

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from AGENT_HARNESS_* environment variables."""

        values: dict[str, object] = {}
        mapping = {
            "MODEL_PROVIDER": "model_provider",
            "OPENAI_API_KEY": "openai_api_key",
            "OPENAI_BASE_URL": "openai_base_url",
            "OPENAI_MODEL": "openai_model",
            "OPENAI_TIMEOUT_SECONDS": "openai_timeout_seconds",
            "DATABASE_PATH": "database_path",
            "MAX_STEPS": "max_steps",
            "TOOL_TIMEOUT_SECONDS": "tool_timeout_seconds",
            "MAX_RETRIES": "max_retries",
            "MAX_INPUT_CHARS": "max_input_chars",
            "MAX_CONTEXT_CHARS": "max_context_chars",
            "MAX_OUTPUT_CHARS": "max_output_chars",
            "LOG_LEVEL": "log_level",
            "OTEL_EXPORTER": "otel_exporter",
            "HOST": "host",
            "PORT": "port",
        }
        integers = {
            "max_steps",
            "max_retries",
            "max_input_chars",
            "max_context_chars",
            "max_output_chars",
            "port",
        }
        floats = {"tool_timeout_seconds", "openai_timeout_seconds"}
        for env_name, field_name in mapping.items():
            raw = os.environ.get(f"AGENT_HARNESS_{env_name}")
            if raw is None or raw == "":
                continue
            if field_name in integers:
                values[field_name] = int(raw)
            elif field_name in floats:
                values[field_name] = float(raw)
            else:
                values[field_name] = raw
        enabled = os.environ.get("AGENT_HARNESS_ENABLE_MCP")
        if enabled is not None and enabled != "":
            values["enable_mcp"] = _bool(enabled)
        allowed = os.environ.get("AGENT_HARNESS_ALLOWED_TOOLS")
        if allowed:
            values["allowed_tools"] = _csv(allowed)
        approval = os.environ.get("AGENT_HARNESS_APPROVAL_TOOLS")
        if approval:
            values["approval_tools"] = _csv(approval)
        return cls.model_validate(values)

    @property
    def checkpoint_path(self) -> str:
        return f"{self.database_path}.checkpoints"
