"""Tests for `MCPAdapter`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.transports import (
    FastMCPTransport,
    PythonStdioTransport,
)
from fastmcp.client.transports.config import MCPConfigTransport

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from pathlib import Path


def _calculator() -> FastMCP[None]:
    server: FastMCP[None] = FastMCP("calc")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return server


@pytest.mark.asyncio
async def test_get_tools_adapts_every_tool_a_server_exposes() -> None:
    server = _calculator()

    @server.tool
    def negate(a: int) -> int:
        """Negate a number."""
        return -a

    tools = await MCPAdapter(server).get_tools()

    assert sorted(tool.name for tool in tools) == ["add", "negate"]


@pytest.mark.asyncio
async def test_tools_work_directly_with_create_agent() -> None:
    agent = create_agent(
        FakeToolCallingModel(
            tool_calls=[[{"name": "add", "args": {"a": 1, "b": 2}, "id": "call-1"}], []]
        ),
        await MCPAdapter(_calculator()).get_tools(),
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "Add 1 and 2."}]})

    assert result["messages"][-1].content.endswith("-3")


@pytest.mark.asyncio
async def test_tools_stay_callable_after_the_adapter_context_exits() -> None:
    """Discovery inside a context must not leave the returned tools dead."""
    async with MCPAdapter(_calculator()) as adapter:
        [tool] = await adapter.get_tools()

    message = await tool.ainvoke(
        {"name": "add", "args": {"a": 2, "b": 3}, "id": "c1", "type": "tool_call"}
    )

    assert message.content[0]["text"] == "5"


def test_target_inference_is_delegated_to_fastmcp(tmp_path: Path) -> None:
    """Targets FastMCP understands are accepted without adapter-side gatekeeping."""
    script = tmp_path / "server.py"
    script.touch()

    assert isinstance(MCPAdapter(script).client.transport, PythonStdioTransport)
    assert isinstance(MCPAdapter(FastMCP("in-process")).client.transport, FastMCPTransport)


def test_one_adapter_can_serve_several_servers() -> None:
    """An MCP config mounts every named server behind a single client."""
    adapter = MCPAdapter(
        {
            "mcpServers": {
                "notes": {"command": "python", "args": ["notes_server.py"]},
                "web": {"url": "https://example.com/mcp"},
            }
        }
    )

    transport = adapter.client.transport
    assert isinstance(transport, MCPConfigTransport)
    assert sorted(transport.config.mcpServers) == ["notes", "web"]


def test_prebuilt_client_is_used_as_is() -> None:
    client: Client[Any] = Client("https://example.com/mcp")

    assert MCPAdapter(client).client is client
