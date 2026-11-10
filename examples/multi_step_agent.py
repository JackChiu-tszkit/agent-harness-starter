"""Example B: two calculator calls, using the deterministic mock model."""

from __future__ import annotations

import sys

from agent_harness.config import Settings
from agent_harness.runtime import build_runtime

REQUEST = "Calculate the total cost of 12 products at 49.90 each, then apply a 15% discount."


def main() -> int:
    runtime = build_runtime(_settings())
    try:
        session = runtime.create_session()
        result = runtime.run(session.id, REQUEST)
    finally:
        runtime.close()
    print(f"status: {result.status}")
    print(f"tools: {','.join(result.tool_calls)}")
    print(f"response: {result.response}")
    if result.status != "completed" or result.response is None or "508.98" not in result.response:
        return 1
    if result.tool_calls != ["calculator", "calculator"]:
        return 1
    return 0


def _settings() -> Settings:
    return Settings(
        model_provider="mock",
        database_path="./data/harness.db",
        otel_exporter="console",
        log_level="INFO",
        enable_mcp=True,
    )


if __name__ == "__main__":
    sys.exit(main())
