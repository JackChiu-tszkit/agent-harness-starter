"""Example C: a remembered fact stays in one session and is invisible to another."""

from __future__ import annotations

import sys

from agent_harness.config import Settings
from agent_harness.runtime import build_runtime

REMEMBER = "Remember that the unit price is 49.90."
QUESTION = "What unit price did I ask you to remember?"


def main() -> int:
    runtime = build_runtime(_settings())
    try:
        session = runtime.create_session()
        remembered = runtime.run(session.id, REMEMBER)
        answer = runtime.run(session.id, QUESTION)
        other = runtime.create_session()
        isolated = runtime.run(other.id, QUESTION)
    finally:
        runtime.close()
    print(f"remember_status: {remembered.status}")
    print(f"remember_response: {remembered.response}")
    print(f"answer_response: {answer.response}")
    print(f"other_session_response: {isolated.response}")
    if remembered.status != "completed" or remembered.response is None:
        return 1
    if "49.90" not in remembered.response or "store_memory" not in remembered.tool_calls:
        return 1
    if answer.status != "completed" or answer.response is None or "49.90" not in answer.response:
        return 1
    if isolated.response is None or "49.90" in isolated.response:
        return 1
    if "do not have a remembered unit price" not in isolated.response:
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
