from agent_harness.memory.store import StateStore


def test_sessions_are_isolated_and_deletable(tmp_path) -> None:
    store = StateStore(str(tmp_path / "harness.db"))
    first = store.create_session()
    second = store.create_session()
    store.add_message(first.id, "user", "only first")
    store.add_memory(first.id, "unit price is 49.90")
    assert store.list_messages(second.id) == []
    assert store.list_memories(second.id) == []
    memory_id = store.list_memories(first.id)[0].id
    assert store.delete_memory(second.id, memory_id) is False
    assert store.list_memories(first.id)[0].content == "unit price is 49.90"
    task_ids = store.delete_session(first.id)
    assert task_ids == []
    assert store.get_session(first.id) is None
    assert store.list_messages(first.id) == []
    assert store.get_session(second.id) is not None
    store.close()

    reopened = StateStore(str(tmp_path / "harness.db"))
    assert reopened.get_session(second.id) is not None
    assert reopened.get_session(first.id) is None
    reopened.close()
