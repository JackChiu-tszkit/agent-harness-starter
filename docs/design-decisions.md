# Design decisions

## One orchestrator

LangGraph owns transitions, checkpoints, and the approval interrupt. The
runtime starts or resumes a graph; it does not choose the next node itself.
A recursion limit is set above `max_steps` so the harness limit fails with an
explicit error first.

## Why the mock model exists

Tests and the three examples must run without a paid API key. `MockModel` is
a deterministic pattern matcher. It is not an LLM and it is not presented as
one. `OpenAICompatibleProvider` is a real HTTP client for
`/chat/completions`. It is unused unless `AGENT_HARNESS_MODEL_PROVIDER=openai`
and the key is present. Provider code stays in `routing/openai.py`.

LangChain Core is installed because current LangGraph depends on it. This
project does not call LangChain APIs.

## SQLite split

Application tables hold sessions, messages, memories, and task status.
LangGraph's `SqliteSaver` holds workflow checkpoints in a sibling file ending
in `.checkpoints`. Deleting a session deletes both. Conversation history is
not the same thing as a checkpoint, and a memory row is not inferred from the
transcript.

## MCP

The official MCP Python SDK 2.3 is used (`MCPServer` and `Client`). FastMCP
was renamed in that release. The example server is safe and deterministic:
calculator, UTC time, and text statistics, plus `unapproved_echo` to show
that discovery is not approval. The default runtime connects in process so
the examples do not depend on a second terminal. `python -m
agent_harness.mcp.server` still serves stdio.

## Approval

Privileged work uses LangGraph `interrupt`. Resume is
`POST /agents/run` with `resume_task_id` and `approve`, so the public route
list stays small. The task must belong to the supplied session.

## Observability

Spans and metrics use the OpenTelemetry SDK. The default exporter prints
spans to the console. Tests use an in-memory exporter. No hosted collector
is required, and an OTLP exporter is not bundled.

## What was left out on purpose

Shell execution, arbitrary file access, and arbitrary network tools are not
implemented. Adding them would need a sandbox this starter does not provide.
Prompt text is not treated as a security boundary.
