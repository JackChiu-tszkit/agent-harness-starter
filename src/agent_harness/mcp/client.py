"""Harness-side MCP client.

Discovery and calls go through the official Python SDK. Only names on the
caller-supplied allowlist are registered. This module does not decide policy.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer

from agent_harness.core.errors import ToolError
from agent_harness.routing.registry import ToolRegistry, ToolSpec

ServerTarget = MCPServer | StdioServerParameters

_IDEMPOTENT = {"calculator", "text_statistics"}


@dataclass
class DiscoveredTool:
    name: str
    description: str
    input_schema: dict


class McpToolClient:
    """Synchronous adapter around the async MCP client."""

    def __init__(self, server: ServerTarget) -> None:
        self._server = server

    def list_tools(self) -> list[DiscoveredTool]:
        return asyncio.run(self._list_tools())

    def call_tool(self, name: str, arguments: dict[str, object], timeout_seconds: float) -> str:
        return asyncio.run(self._call_tool(name, arguments, timeout_seconds))

    async def _list_tools(self) -> list[DiscoveredTool]:
        async with Client(self._server) as client:
            listed = await client.list_tools()
        return [
            DiscoveredTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.input_schema),
            )
            for tool in listed.tools
        ]

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        timeout_seconds: float,
    ) -> str:
        try:
            async with Client(self._server, read_timeout_seconds=timeout_seconds) as client:
                result = await client.call_tool(
                    name,
                    arguments,
                    read_timeout_seconds=timeout_seconds,
                )
        except Exception as exc:
            raise ToolError(
                f"MCP connection failed ({type(exc).__name__}).",
                transient=True,
            ) from exc
        text = result.content[0].text if result.content else ""
        if result.is_error:
            raise ToolError(text or f"MCP tool '{name}' failed.", transient=False)
        return text


def register_approved_mcp_tools(
    registry: ToolRegistry,
    client: McpToolClient,
    allowed: set[str],
    timeout_seconds: float,
) -> list[str]:
    """Register discovered tools whose names are explicitly approved."""

    registered: list[str] = []
    for tool in client.list_tools():
        if tool.name not in allowed:
            continue
        registry.register(
            ToolSpec(
                name=tool.name,
                description=tool.description,
                idempotent=tool.name in _IDEMPOTENT,
                source="mcp",
                handler=_mcp_handler(client, tool.name, timeout_seconds),
                schema=tool.input_schema,
            ),
            replace=True,
        )
        registered.append(tool.name)
    return registered


def _mcp_handler(client: McpToolClient, name: str, timeout_seconds: float):
    def handler(arguments: dict[str, object], _context: object) -> str:
        return client.call_tool(name, arguments, timeout_seconds)

    return handler
