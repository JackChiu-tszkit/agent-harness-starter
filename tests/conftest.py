from __future__ import annotations

import pytest

from agent_harness.config import Settings
from agent_harness.runtime import Runtime, build_runtime


def make_settings(tmp_path, **overrides) -> Settings:
    values = {
        "model_provider": "mock",
        "database_path": str(tmp_path / "harness.db"),
        "otel_exporter": "memory",
        "log_level": "INFO",
        "enable_mcp": False,
        "tool_timeout_seconds": 2,
        "max_steps": 8,
        "max_retries": 2,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def runtime(settings) -> Runtime:
    harness = build_runtime(settings)
    try:
        yield harness
    finally:
        harness.close()
