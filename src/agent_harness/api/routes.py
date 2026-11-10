"""HTTP routes. Handlers are synchronous because tool calls use asyncio.run."""

from __future__ import annotations

import re
import uuid

from fastapi import APIRouter, Header, Request, Response

from agent_harness.api.schemas import (
    HealthResponse,
    MessageView,
    RunRequest,
    RunResponse,
    SessionCreated,
    SessionView,
    TaskView,
)
from agent_harness.core.errors import HarnessError
from agent_harness.core.models import RunResult

router = APIRouter()
_CORRELATION = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/sessions", response_model=SessionCreated, status_code=201)
def create_session(request: Request) -> SessionCreated:
    session = request.app.state.runtime.create_session()
    return SessionCreated(
        session_id=session.id,
        status=session.status,
        created_at=session.created_at,
    )


@router.get("/sessions/{session_id}", response_model=SessionView)
def get_session(session_id: str, request: Request) -> SessionView:
    store = request.app.state.runtime.store
    session = store.get_session(session_id)
    if session is None:
        raise HarnessError("session_not_found", "session was not found", 404)
    messages = store.list_messages(session_id)
    memories = [item.content for item in store.list_memories(session_id)]
    return SessionView(
        session_id=session.id,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[
            MessageView(role=item.role, content=item.content, created_at=item.created_at)
            for item in messages
        ],
        memories=memories,
    )


@router.post("/agents/run", response_model=RunResponse)
def run_agent(
    body: RunRequest,
    request: Request,
    response: Response,
    x_correlation_id: str | None = Header(default=None),
) -> RunResponse:
    runtime = request.app.state.runtime
    correlation_id = None
    if x_correlation_id and _CORRELATION.fullmatch(x_correlation_id):
        correlation_id = x_correlation_id
    if body.resume_task_id:
        result = runtime.resume(
            body.session_id,
            body.resume_task_id,
            approved=bool(body.approve),
            correlation_id=correlation_id,
        )
    else:
        result = runtime.run(
            body.session_id,
            body.message or "",
            correlation_id or str(uuid.uuid4()),
        )
    response.headers["X-Correlation-ID"] = result.correlation_id
    return _run_response(result)


@router.get("/tasks/{task_id}", response_model=TaskView)
def get_task(task_id: str, request: Request) -> TaskView:
    task = request.app.state.runtime.store.get_task(task_id)
    if task is None:
        raise HarnessError("task_not_found", "task was not found", 404)
    return TaskView(
        task_id=task.id,
        session_id=task.session_id,
        status=task.status,
        response=task.response,
        error=task.error,
        step_count=task.step_count,
        correlation_id=task.correlation_id,
        tool_calls=task.tool_calls,
        approval=task.approval,
    )


def _run_response(result: RunResult) -> RunResponse:
    return RunResponse(
        task_id=result.task_id,
        session_id=result.session_id,
        status=result.status,
        response=result.response,
        error=result.error,
        step_count=result.step_count,
        correlation_id=result.correlation_id,
        tool_calls=result.tool_calls,
        approval=result.approval,
    )
