"""Run the local API server."""

from __future__ import annotations

import uvicorn

from agent_harness.config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "agent_harness.api.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
