"""FastAPI application.

Authentication and multi-tenant authorization are not implemented. Anyone who
can reach this process and knows an id can read that session or task.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from agent_harness import __version__
from agent_harness.api.routes import router
from agent_harness.api.schemas import ErrorResponse
from agent_harness.config import Settings
from agent_harness.core.errors import HarnessError
from agent_harness.core.models import ModelProvider
from agent_harness.routing.registry import ToolSpec
from agent_harness.runtime.service import build_runtime


def create_app(
    settings: Settings | None = None,
    *,
    model: ModelProvider | None = None,
    extra_tools: list[ToolSpec] | None = None,
) -> FastAPI:
    resolved = settings or Settings.from_env()
    runtime = build_runtime(resolved, model=model, extra_tools=extra_tools)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.runtime.close()

    app = FastAPI(
        title="Agent Harness Starter",
        version=__version__,
        summary="Local agent harness API.",
        description=(
            "A local demonstration API for the agent harness. "
            "Authentication and multi-tenant authorization are not implemented. "
            "Do not expose this server to an untrusted network."
        ),
        lifespan=lifespan,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    )
    app.state.runtime = runtime

    @app.exception_handler(HarnessError)
    def handle_harness_error(_request, exc: HarnessError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "detail": exc.message},
        )

    app.include_router(router)
    return app


app = create_app()
