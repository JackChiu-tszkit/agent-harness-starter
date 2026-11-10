"""The only workflow orchestrator.

LangGraph chooses the next node, stores checkpoints, and pauses for approval.
Do not add a second loop that decides node order.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from agent_harness.core.errors import HarnessError, OutputError, ToolError, ToolTimeoutError
from agent_harness.core.models import ChatMessage, ModelProvider, ModelRequest
from agent_harness.memory.store import StateStore
from agent_harness.observability.logging import LogContext
from agent_harness.observability.tracing import Telemetry
from agent_harness.orchestration.state import AgentState
from agent_harness.prompts.policy import ExecutionPolicy
from agent_harness.routing.registry import ToolContext, ToolRegistry
from agent_harness.safety.guardrails import validate_output, wrap_tool_output

_LOG = logging.getLogger("agent_harness.workflow")


@dataclass
class GraphDeps:
    model: ModelProvider
    registry: ToolRegistry
    policy: ExecutionPolicy
    store: StateStore
    telemetry: Telemetry


def build_graph(deps: GraphDeps, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    def prepare(state: AgentState) -> AgentState:
        transcript = list(state.get("history") or [])
        transcript.append({"role": "user", "content": state["user_message"]})
        return {
            "transcript": transcript,
            "status": "running",
            "step_count": 0,
            "tool_calls": [],
            "pending_tool": None,
            "response": None,
            "error": None,
        }

    def route(state: AgentState) -> AgentState:
        log = _log(state)
        if deps.store.is_cancelled(state["task_id"]):
            log.info("task cancelled")
            return {"status": "cancelled", "error": "Task was cancelled.", "pending_tool": None}
        step = int(state.get("step_count") or 0) + 1
        limit = int(state["max_steps"])
        if step > limit:
            deps.telemetry.errors.add(1, {"status": "max_steps"})
            return {
                "step_count": step,
                "status": "failed",
                "error": f"Exceeded maximum step count ({limit}).",
                "pending_tool": None,
            }
        try:
            with deps.telemetry.tracer.start_as_current_span("agent.model") as span:
                span.set_attribute("task.id", state["task_id"])
                span.set_attribute("model.name", deps.model.name)
                response = deps.model.complete(_request(deps, state))
        except HarnessError as exc:
            log.info("model failed error_type=HarnessError")
            return {
                "step_count": step,
                "status": "failed",
                "error": exc.message,
                "pending_tool": None,
            }
        except Exception:
            log.info("model failed error_type=Exception")
            return {
                "step_count": step,
                "status": "failed",
                "error": "The model provider failed.",
                "pending_tool": None,
            }
        if response.tool_calls:
            call = response.tool_calls[0]
            log.info("model selected tool name=%s", call.name)
            return {
                "step_count": step,
                "status": "running",
                "pending_tool": {"name": call.name, "arguments": call.arguments},
                "response": None,
            }
        if response.content:
            return {
                "step_count": step,
                "status": "running",
                "pending_tool": None,
                "response": response.content,
            }
        return {
            "step_count": step,
            "status": "failed",
            "error": "The model returned an empty response.",
            "pending_tool": None,
        }

    def guard(state: AgentState) -> AgentState:
        pending = state.get("pending_tool") or {}
        name = str(pending.get("name") or "")
        spec = deps.registry.get(name)
        if name not in deps.policy.allowed_tools:
            return _denied(f"Tool '{name}' is not allowlisted.")
        if spec is None:
            return _denied(f"Tool '{name}' is not available.")
        try:
            validated = spec.validate(pending.get("arguments") or {})
        except ToolError as exc:
            return _denied(str(exc))
        if name in deps.policy.approval_tools:
            decision = interrupt(
                {
                    "tool": name,
                    "arguments": validated,
                    "reason": "This privileged tool requires human approval before it can run.",
                }
            )
            if not isinstance(decision, dict) or decision.get("approved") is not True:
                return _denied(f"Approval denied for tool '{name}'.")
        return {"pending_tool": {"name": name, "arguments": validated}, "status": "running"}

    def execute(state: AgentState) -> AgentState:
        pending = state.get("pending_tool") or {}
        name = str(pending.get("name") or "")
        spec = deps.registry.get(name)
        step = int(state.get("step_count") or 0) + 1
        limit = int(state["max_steps"])
        if spec is None:
            return _denied(f"Tool '{name}' is not available.", step=step)
        if step > limit:
            deps.telemetry.errors.add(1, {"status": "max_steps"})
            return {
                "step_count": step,
                "status": "failed",
                "error": f"Exceeded maximum step count ({limit}).",
                "pending_tool": None,
            }
        arguments = dict(pending.get("arguments") or {})
        context = ToolContext(state["session_id"], state["task_id"], deps.store)
        attempts = 1 + (deps.policy.max_retries if spec.idempotent else 0)
        last_error = "tool failed"
        for _attempt in range(attempts):
            try:
                with deps.telemetry.tracer.start_as_current_span("agent.tool") as span:
                    span.set_attribute("tool.name", name)
                    span.set_attribute("tool.source", spec.source)
                    span.set_attribute("task.id", state["task_id"])
                    output = _run_bounded(
                        lambda: spec.handler(arguments, context),
                        deps.policy.tool_timeout_seconds,
                        name,
                    )
                deps.telemetry.tool_calls.add(1, {"tool": name, "status": "ok"})
                _log(state).info("tool finished name=%s status=ok", name)
                return {
                    "step_count": step,
                    "status": "running",
                    "tool_output": output,
                    "tool_error": None,
                }
            except ToolTimeoutError as exc:
                last_error = str(exc)
                deps.telemetry.tool_calls.add(1, {"tool": name, "status": "timeout"})
                _log(state).info("tool finished name=%s status=timeout", name)
                if not spec.idempotent:
                    break
            except ToolError as exc:
                last_error = str(exc)
                deps.telemetry.tool_calls.add(1, {"tool": name, "status": "error"})
                _log(state).info("tool finished name=%s status=error", name)
                if not (exc.transient and spec.idempotent):
                    break
        return {
            "step_count": step,
            "status": "running",
            "tool_output": None,
            "tool_error": last_error,
        }

    def observe(state: AgentState) -> AgentState:
        pending = state.get("pending_tool") or {}
        name = str(pending.get("name") or "")
        if state.get("tool_output") is not None:
            payload = str(state["tool_output"])
        else:
            payload = f"Tool error: {state.get('tool_error') or 'tool failed'}"
        transcript = list(state.get("transcript") or [])
        transcript.append(
            {
                "role": "tool",
                "content": wrap_tool_output(name, payload),
                "tool_name": name,
            }
        )
        calls = list(state.get("tool_calls") or [])
        calls.append(name)
        return {
            "transcript": transcript,
            "tool_calls": calls,
            "pending_tool": None,
            "tool_output": None,
            "tool_error": None,
            "status": "running",
        }

    def finalize(state: AgentState) -> AgentState:
        try:
            text = validate_output(state.get("response"), deps.policy.max_output_chars)
        except OutputError as exc:
            return {"status": "failed", "error": exc.message, "response": None}
        _log(state).info("task completed")
        return {"status": "completed", "response": text, "error": None, "pending_tool": None}

    def fail(state: AgentState) -> AgentState:
        status = state.get("status")
        if status not in {"failed", "cancelled"}:
            status = "failed"
        _log(state).info("task stopped status=%s", status)
        return {
            "status": status,
            "error": state.get("error") or "The task failed.",
            "pending_tool": None,
        }

    builder = StateGraph(AgentState)
    builder.add_node("prepare", prepare)
    builder.add_node("route", route)
    builder.add_node("guard", guard)
    builder.add_node("execute", execute)
    builder.add_node("observe", observe)
    builder.add_node("finalize", finalize)
    builder.add_node("fail", fail)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "route")
    builder.add_conditional_edges(
        "route",
        _after_route,
        {"guard": "guard", "finalize": "finalize", "fail": "fail"},
    )
    builder.add_conditional_edges("guard", _after_guard, {"execute": "execute", "fail": "fail"})
    builder.add_conditional_edges(
        "execute",
        _after_execute,
        {"observe": "observe", "fail": "fail"},
    )
    builder.add_edge("observe", "route")
    builder.add_edge("finalize", END)
    builder.add_edge("fail", END)
    return builder.compile(checkpointer=checkpointer)


def _request(deps: GraphDeps, state: AgentState) -> ModelRequest:
    messages = [ChatMessage.model_validate(item) for item in state.get("transcript") or []]
    return ModelRequest(
        system_prompt=state["system_prompt"],
        messages=messages,
        memories=list(state.get("memories") or []),
        tools=deps.registry.visible(deps.policy.allowed_tools),
    )


def _after_route(state: AgentState) -> str:
    if state.get("status") in {"failed", "cancelled"}:
        return "fail"
    if state.get("pending_tool"):
        return "guard"
    return "finalize"


def _after_guard(state: AgentState) -> str:
    if state.get("status") in {"failed", "cancelled"}:
        return "fail"
    return "execute"


def _after_execute(state: AgentState) -> str:
    if state.get("status") in {"failed", "cancelled"}:
        return "fail"
    return "observe"


def _denied(message: str, step: int | None = None) -> AgentState:
    update: AgentState = {"status": "failed", "error": message, "pending_tool": None}
    if step is not None:
        update["step_count"] = step
    return update


def _log(state: AgentState) -> LogContext:
    return LogContext(
        _LOG,
        {
            "correlation_id": state.get("correlation_id"),
            "session_id": state.get("session_id"),
            "task_id": state.get("task_id"),
        },
    )


def _run_bounded(func, timeout_seconds: float, tool_name: str) -> str:
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["value"] = func()
        except Exception as exc:  # noqa: BLE001 - transferred to the caller thread
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise ToolTimeoutError(f"Tool '{tool_name}' timed out after {timeout_seconds} seconds.")
    if "error" in box:
        error = box["error"]
        if isinstance(error, Exception):
            raise error
        raise ToolError(f"Tool '{tool_name}' failed.")
    value = box.get("value")
    if not isinstance(value, str):
        raise ToolError(f"Tool '{tool_name}' returned an invalid result.")
    return value
