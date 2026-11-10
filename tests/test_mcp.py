import sys

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from agent_harness.core.errors import ToolError
from agent_harness.mcp.client import McpToolClient
from agent_harness.mcp.server import call_counts, create_server


def test_in_process_mcp_success_and_failure() -> None:
    before = call_counts["calculator"]
    client = McpToolClient(create_server())
    names = [tool.name for tool in client.list_tools()]
    assert {"calculator", "current_utc_time", "text_statistics", "unapproved_echo"} <= set(names)
    assert client.call_tool("calculator", {"expression": "125 * 48"}, timeout_seconds=5) == "6000"
    assert call_counts["calculator"] == before + 1
    with pytest.raises(ToolError, match="division by zero"):
        client.call_tool("calculator", {"expression": "1 / 0"}, timeout_seconds=5)
    with pytest.raises(ToolError, match="Unknown tool"):
        client.call_tool("missing", {}, timeout_seconds=5)


def test_stdio_mcp_success_and_failure() -> None:
    import asyncio

    async def scenario() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agent_harness.mcp.server"],
        )
        async with Client(params) as client:
            success = await client.call_tool("calculator", {"expression": "2 + 2"})
            failure = await client.call_tool("calculator", {"expression": "1 / 0"})
            missing = await client.call_tool("missing", {})
        assert success.is_error is False
        assert success.content[0].text == "4"
        assert failure.is_error is True
        assert missing.is_error is True

    asyncio.run(scenario())
