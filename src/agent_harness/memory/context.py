"""Short-term context assembly. This does not store a second copy of history."""

from __future__ import annotations

from agent_harness.core.models import MessageRecord, SessionContext
from agent_harness.memory.store import StateStore


def trim_messages(messages: list[MessageRecord], max_chars: int) -> list[MessageRecord]:
    """Keep the newest suffix that fits in max_chars."""

    kept: list[MessageRecord] = []
    total = 0
    for message in reversed(messages):
        size = len(message.content)
        if kept and total + size > max_chars:
            break
        if not kept and size > max_chars:
            break
        kept.append(message)
        total += size
    return list(reversed(kept))


def assemble_context(store: StateStore, session_id: str, max_chars: int) -> SessionContext:
    history = [
        {"role": message.role, "content": message.content}
        for message in trim_messages(store.list_messages(session_id), max_chars)
    ]
    memories = [item.content for item in store.list_memories(session_id)]
    return SessionContext(session_id=session_id, history=history, memories=memories)
