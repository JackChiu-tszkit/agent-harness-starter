from fastapi.testclient import TestClient

from agent_harness.api.app import create_app
from tests.conftest import make_settings


def test_api_session_isolation_and_openapi(tmp_path) -> None:
    app = create_app(make_settings(tmp_path, enable_mcp=True))
    with TestClient(app) as client:
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        paths = schema.json()["paths"]
        expected = {
            "/health",
            "/sessions",
            "/agents/run",
            "/sessions/{session_id}",
            "/tasks/{task_id}",
        }
        assert expected <= set(paths)
        assert "not implemented" in schema.json()["info"]["description"]

        assert client.get("/health").json() == {"status": "ok"}
        created = client.post("/sessions")
        assert created.status_code == 201
        session_id = created.json()["session_id"]
        other = client.post("/sessions").json()["session_id"]

        rejected = client.post("/agents/run", json={"session_id": session_id, "message": "   "})
        assert rejected.status_code == 422

        run = client.post(
            "/agents/run",
            headers={"X-Correlation-ID": "trace-123"},
            json={"session_id": session_id, "message": "What is 125 multiplied by 48?"},
        )
        assert run.status_code == 200
        body = run.json()
        assert body["status"] == "completed"
        assert body["response"] == "125 multiplied by 48 is 6000."
        assert body["tool_calls"] == ["calculator"]
        assert run.headers["X-Correlation-ID"] == "trace-123"

        task = client.get(f"/tasks/{body['task_id']}")
        assert task.status_code == 200
        assert task.json()["session_id"] == session_id

        view = client.get(f"/sessions/{session_id}")
        other_view = client.get(f"/sessions/{other}")
        assert "125 multiplied by 48" in view.text
        assert "125 multiplied by 48" not in other_view.text
        assert client.get("/sessions/missing-session").status_code == 404
        assert client.get("/tasks/missing-task").status_code == 404
        assert "calculator" in app.state.runtime.mcp_tools
        assert "unapproved_echo" not in app.state.runtime.registry.names()
