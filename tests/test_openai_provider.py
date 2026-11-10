import json

import httpx
import pytest

from agent_harness.core.errors import HarnessError
from agent_harness.core.models import ModelRequest, ToolDefinition
from agent_harness.routing.openai import OpenAICompatibleProvider


def _request() -> ModelRequest:
    return ModelRequest(
        system_prompt="Prompt-Version: v1\ntrusted instructions",
        messages=[],
        memories=["unit price is 49.90"],
        tools=[
            ToolDefinition(
                name="calculator",
                description="add",
                parameters={"type": "object", "properties": {}},
            )
        ],
    )


def test_openai_provider_parses_a_tool_call_and_hides_the_key() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["authorization"]
        body = json.loads(request.content.decode())
        seen["body"] = body
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression": "1 + 1"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    provider = OpenAICompatibleProvider(
        api_key="sk-test-secret",
        base_url="https://example.invalid/v1",
        model="gpt-test",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )
    response = provider.complete(_request())
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"expression": "1 + 1"}
    assert seen["authorization"] == "Bearer sk-test-secret"
    system = seen["body"]["messages"][0]["content"]
    assert "unit price is 49.90" not in system
    assert any("unit price is 49.90" in message["content"] for message in seen["body"]["messages"])


def test_openai_provider_error_does_not_echo_the_key() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key sk-test-secret"}})

    provider = OpenAICompatibleProvider(
        api_key="sk-test-secret",
        base_url="https://example.invalid/v1",
        model="gpt-test",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(HarnessError) as caught:
        provider.complete(_request())
    assert caught.value.status_code == 502
    assert "sk-test-secret" not in caught.value.message
    assert "HTTP 401" in caught.value.message
