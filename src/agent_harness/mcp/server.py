"""Example MCP server.

The server exposes safe, deterministic tools. It does not read the harness
database and it does not receive session credentials. Run it on stdio with
`python -m agent_harness.mcp.server`.
"""

from __future__ import annotations

import logging
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as McpToolError

from agent_harness.core.errors import ToolError
from agent_harness.tools.builtin import current_utc_time, text_statistics
from agent_harness.tools.calculator import evaluate_expression

call_counts = {
    "calculator": 0,
    "current_utc_time": 0,
    "text_statistics": 0,
    "unapproved_echo": 0,
}


def create_server() -> MCPServer:
    server = MCPServer("agent-harness-tools")

    @server.tool(
        description="Evaluate a basic arithmetic expression using only numbers and + - * /."
    )
    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression."""

        call_counts["calculator"] += 1
        try:
            return evaluate_expression(expression)
        except ToolError as exc:
            raise McpToolError(str(exc)) from exc

    @server.tool(
        name="current_utc_time",
        description="Return the current UTC time in ISO 8601 format.",
    )
    def current_utc_time_tool() -> str:
        """Return the current UTC time."""

        call_counts["current_utc_time"] += 1
        return current_utc_time()

    @server.tool(
        name="text_statistics",
        description="Count characters, words, and lines in a text value.",
    )
    def text_statistics_tool(text: str) -> str:
        """Return character, word, and line counts."""

        call_counts["text_statistics"] += 1
        return text_statistics(text)

    @server.tool(description="Echo text. The default harness allowlist does not approve this tool.")
    def unapproved_echo(text: str) -> str:
        """Echo text. Not approved for the agent by default."""

        call_counts["unapproved_echo"] += 1
        return text

    return server


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
