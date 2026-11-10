# Security

This starter is a local demonstration. It is not an authenticated multi-tenant
service. Do not expose the API to an untrusted network. A caller who knows a
session id or task id can read that record. UUIDs make guessing harder; they
are not authorization.

## Controls that are implemented

- Empty and oversized inputs are rejected.
- Tool arguments are validated before the handler runs.
- The model only sees allowlisted tools, and the guard checks the allowlist again.
- `delete_memory` is privileged. It is off the default allowlist. If it is
  allowlisted, LangGraph interrupts until `POST /agents/run` resumes the task
  with `approve: true`.
- A tool can delete only a memory row whose `session_id` matches the running
  session.
- Tool execution has a timeout and a maximum step count.
- Transient failures retry only when the tool is marked idempotent, and only
  up to `max_retries`.
- Tool output is wrapped as untrusted data. A small set of prompt-injection
  phrases is flagged. This does not make the model safe.
- Logs and span attributes do not include the user message, tool arguments, or
  API keys. The log formatter redacts common credential shapes.
- The calculator accepts only numbers and `+ - * /`. Names, calls, and
  exponentiation are rejected.
- There is no shell tool, no general filesystem tool, and no general HTTP tool.

## Controls that are not implemented

- Authentication, authorization, and tenant isolation.
- A hard stop of a tool thread after a timeout. The caller stops waiting; a
  daemon thread may continue until it returns.
- A guarantee that a model will ignore instructions hidden in tool output.
- Network egress control beyond the choice not to ship an open network tool.
- Encryption of the SQLite files.

## Retention

`Runtime.delete_session(session_id)` deletes that session's messages,
memories, tasks, and LangGraph checkpoints. `Runtime.close_session` rejects
new runs but keeps the rows until deletion. The HTTP API does not expose
deletion. Call the runtime method from Python, or delete the database file
when the process is stopped.
