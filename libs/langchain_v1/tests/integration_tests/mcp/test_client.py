"""Integration tests for `langchain.mcp` against a real stdio MCP server."""

from __future__ import annotations

import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from langchain.mcp import load_mcp_tools

_SERVER_SCRIPT = str(Path(__file__).parent / "_server.py")

_ADD_CALL = {"type": "tool_call", "name": "add", "args": {"a": 2, "b": 3}, "id": "1"}


def _client() -> Client[StdioTransport]:
    return Client(StdioTransport(command=sys.executable, args=[_SERVER_SCRIPT]))


async def test_load_mcp_tools_lists_server_tools() -> None:
    async with _client() as client:
        tools = await load_mcp_tools(client)
    assert {"add", "greet"} <= {tool.name for tool in tools}


async def test_call_tool_end_to_end() -> None:
    async with _client() as client:
        tools = {tool.name: tool for tool in await load_mcp_tools(client)}
        message = await tools["add"].ainvoke(_ADD_CALL)
    assert "5" in str(message.content)
