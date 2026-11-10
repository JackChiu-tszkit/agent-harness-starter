"""Execute one LangGraph task and persist its outcome."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent_harness.config import Settings
from agent_harness.core.errors import HarnessError
from agent_harness.core.models import ModelProvider, RunResult
from agent_harness.mcp.client import McpToolClient, register_approved_mcp_tools
from agent_harness.mcp.server import create_server
from agent_harness.memory.context import assemble_context
from agent_harness.memory.store import StateStore
from agent_harness.observability.logging import LogContext, configure_logging
from agent_harness.observability.tracing import Telemetry
from agent_harness.orchestration.graph import GraphDeps, build_graph
from agent_harness.prompts.policy import ExecutionPolicy, policy_from_settings
from agent_harness.prompts.templates import build_system_prompt, select_task
from agent_harness.routing.mock import MockModel
from agent_harness.routing.openai import OpenAICompatibleProvider
from agent_harness.routing.registry import ToolRegistry, ToolSpec
from agent_harness.runtime.local_tools import build_local_registry
from agent_harness.safety.guardrails import validate_input


class Runtime:
    def __init__(
        self,
        settings: Settings,
        store: StateStore,
        registry: ToolRegistry,
        policy: ExecutionPolicy,
        telemetry: Telemetry,
        graph,
        checkpoint_context,
        mcp_tools: list[str],
        mcp_error: str | None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.registry = registry
        self.policy = policy
        self.telemetry = telemetry
        self.graph = graph
        self.mcp_tools = mcp_tools
        self.mcp_error = mcp_error
        self._checkpoint_context = checkpoint_context
        self._closed = False

    def create_session(self):
        return self.store.create_session()

    def close_session(self, session_id: str):
        return self.store.close_session(session_id)

    def delete_session(self, session_id: str) -> None:
        task_ids = self.store.delete_session(session_id)
        for task_id in task_ids:
            self.graph.checkpointer.delete_thread(task_id)

    def run(self, session_id: str, message: str, correlation_id: str | None = None) -> RunResult:
        started = time.perf_counter()
        session = self.store.require_active_session(session_id)
        cleaned = validate_input(message, self.policy.max_input_chars)
        correlation = correlation_id or str(uuid.uuid4())
        task = self.store.create_task(session.id, cleaned, correlation)
        context = assemble_context(self.store, session.id, self.policy.max_context_chars)
        self.store.add_message(session.id, "user", cleaned)
        state = {
            "session_id": session.id,
            "task_id": task.id,
            "correlation_id": correlation,
            "user_message": cleaned,
            "system_prompt": build_system_prompt(select_task(cleaned)),
            "memories": context.memories,
            "history": context.history,
            "max_steps": self.policy.max_steps,
        }
        log = LogContext(
            configure_logging(self.settings.log_level),
            {"correlation_id": correlation, "session_id": session.id, "task_id": task.id},
        )
        log.info("task started message_length=%s", len(cleaned))
        status = "failed"
        try:
            with self.telemetry.tracer.start_as_current_span("agent.task") as span:
                _span_ids(span, session.id, task.id, correlation)
                self.graph.invoke(state, _config(task.id, self.policy.max_steps))
            result = self._result_from_checkpoint(task.id, session.id, correlation, cleaned)
            status = result.status
            return result
        except HarnessError:
            raise
        except Exception as exc:
            self.telemetry.errors.add(1, {"status": "workflow_error"})
            log.info("workflow failed error_type=%s", type(exc).__name__)
            self.store.update_task(
                task.id,
                status="failed",
                error="The workflow failed.",
                step_count=0,
            )
            raise HarnessError("workflow_error", "The workflow failed.", 500) from exc
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.telemetry.task_duration.record(elapsed_ms, {"status": status})

    def resume(
        self,
        session_id: str,
        task_id: str,
        *,
        approved: bool,
        correlation_id: str | None = None,
    ) -> RunResult:
        started = time.perf_counter()
        self.store.require_active_session(session_id)
        task = self.store.require_task(task_id, session_id)
        if task.status != "awaiting_approval":
            raise HarnessError("task_not_resumable", "task is not waiting for approval", 409)
        correlation = task.correlation_id or correlation_id or str(uuid.uuid4())
        status = "failed"
        try:
            with self.telemetry.tracer.start_as_current_span("agent.task") as span:
                _span_ids(span, session_id, task_id, correlation)
                self.graph.invoke(
                    Command(resume={"approved": approved}),
                    _config(task_id, self.policy.max_steps),
                )
            result = self._result_from_checkpoint(task_id, session_id, correlation, task.request)
            status = result.status
            return result
        except HarnessError:
            raise
        except Exception as exc:
            self.telemetry.errors.add(1, {"status": "workflow_error"})
            self.store.update_task(task_id, status="failed", error="The workflow failed.")
            raise HarnessError("workflow_error", "The workflow failed.", 500) from exc
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.telemetry.task_duration.record(elapsed_ms, {"status": status})

    def request_cancel(self, task_id: str) -> None:
        if self.store.get_task(task_id) is None:
            raise HarnessError("task_not_found", "task was not found", 404)
        self.store.request_cancel(task_id)

    def workflow_checkpoint_count(self, task_id: str) -> int:
        config = _config(task_id, self.policy.max_steps)
        return sum(1 for _ in self.graph.get_state_history(config))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.telemetry.shutdown()
        self._checkpoint_context.__exit__(None, None, None)
        self.store.close()

    def _result_from_checkpoint(
        self,
        task_id: str,
        session_id: str,
        correlation_id: str,
        request: str,
    ) -> RunResult:
        snapshot = self.graph.get_state(_config(task_id, self.policy.max_steps))
        if snapshot.interrupts:
            raw = snapshot.interrupts[0].value
            approval = raw if isinstance(raw, dict) else {"reason": "approval required"}
            values = snapshot.values or {}
            step_count = int(values.get("step_count") or 0)
            tool_calls = list(values.get("tool_calls") or [])
            self.store.update_task(
                task_id,
                status="awaiting_approval",
                step_count=step_count,
                tool_calls=tool_calls,
                approval=approval,
            )
            return RunResult(
                task_id=task_id,
                session_id=session_id,
                status="awaiting_approval",
                response=None,
                error=None,
                step_count=step_count,
                correlation_id=correlation_id,
                tool_calls=tool_calls,
                approval=approval,
            )
        values = snapshot.values or {}
        status = str(values.get("status") or "failed")
        response = values.get("response")
        error = values.get("error")
        step_count = int(values.get("step_count") or 0)
        tool_calls = list(values.get("tool_calls") or [])
        if status == "completed" and isinstance(response, str):
            self.store.add_message(session_id, "assistant", response)
        elif status in {"failed", "cancelled"}:
            self.telemetry.errors.add(1, {"status": status})
        self.store.update_task(
            task_id,
            status=status,
            response=response if isinstance(response, str) else None,
            error=error if isinstance(error, str) else None,
            step_count=step_count,
            tool_calls=tool_calls,
        )
        return RunResult(
            task_id=task_id,
            session_id=session_id,
            status=status,
            response=response if isinstance(response, str) else None,
            error=error if isinstance(error, str) else None,
            step_count=step_count,
            correlation_id=correlation_id,
            tool_calls=tool_calls,
        )


def build_runtime(
    settings: Settings,
    *,
    model: ModelProvider | None = None,
    extra_tools: list[ToolSpec] | None = None,
) -> Runtime:
    logger = configure_logging(settings.log_level)
    store = StateStore(settings.database_path)
    registry = build_local_registry()
    policy = policy_from_settings(settings)
    telemetry = Telemetry(settings.otel_exporter)
    mcp_tools: list[str] = []
    mcp_error: str | None = None
    if settings.enable_mcp:
        try:
            client = McpToolClient(create_server())
            mcp_tools = register_approved_mcp_tools(
                registry,
                client,
                set(settings.allowed_tools),
                settings.tool_timeout_seconds,
            )
            logger.info("mcp tools registered count=%s", len(mcp_tools))
        except Exception as exc:
            mcp_error = type(exc).__name__
            logger.warning("mcp discovery failed error_type=%s", mcp_error)
    for spec in extra_tools or []:
        registry.register(spec, replace=True)
    selected = model or _provider(settings)
    checkpoint_context = SqliteSaver.from_conn_string(settings.checkpoint_path)
    try:
        checkpointer = checkpoint_context.__enter__()
        graph = build_graph(
            GraphDeps(selected, registry, policy, store, telemetry),
            checkpointer,
        )
    except Exception:
        checkpoint_context.__exit__(None, None, None)
        store.close()
        telemetry.shutdown()
        raise
    return Runtime(
        settings,
        store,
        registry,
        policy,
        telemetry,
        graph,
        checkpoint_context,
        mcp_tools,
        mcp_error,
    )


def _provider(settings: Settings) -> ModelProvider:
    if settings.model_provider == "mock":
        return MockModel()
    if not settings.openai_api_key:
        raise HarnessError(
            "configuration_error",
            "AGENT_HARNESS_OPENAI_API_KEY is required for the openai provider.",
            500,
        )
    return OpenAICompatibleProvider(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
    )


def _config(task_id: str, max_steps: int) -> dict:
    return {
        "configurable": {"thread_id": task_id},
        "recursion_limit": max(50, max_steps * 4),
    }


def _span_ids(span, session_id: str, task_id: str, correlation_id: str) -> None:
    span.set_attribute("session.id", session_id)
    span.set_attribute("task.id", task_id)
    span.set_attribute("correlation.id", correlation_id)


def iter_checkpoint_tasks(runtime: Runtime, task_id: str) -> Iterator[object]:
    return runtime.graph.get_state_history(_config(task_id, runtime.policy.max_steps))
