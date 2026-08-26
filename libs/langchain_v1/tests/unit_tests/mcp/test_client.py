from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.transports import (
    FastMCPTransport,
    PythonStdioTransport,
)
from fastmcp.client.transports.config import MCPConfigTransport
from mcp.types import CallToolResult, TextContent, Tool

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from pathlib import Path


class FakeClient:
    def __init__(self, tools: list[Tool], result: CallToolResult) -> None:
        self.tools = tools
        self.result = result
        self.enter_count = 0
        self.exit_count = 0
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def __aenter__(self) -> Self:
        self.enter_count += 1
        return self

    async def __aexit__(self, *_: object) -> None:
        self.exit_count += 1

    async def list_tools(self) -> list[Tool]:
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        self.calls.append((name, arguments))
        return self.result


def make_adapter(client: FakeClient) -> MCPAdapter:
    adapter = MCPAdapter("https://example.com/mcp")
    adapter._client = client  # type: ignore[assignment]
    return adapter


@pytest.mark.asyncio
async def test_get_tools_converts_schema_metadata_and_results() -> None:
    remote_tool = Tool(
        name="add",
        title="Add numbers",
        description="Add two numbers together.",
        inputSchema={"properties": {"a": {"type": "integer"}}},
    )
    client = FakeClient(
        [remote_tool],
        CallToolResult(content=[TextContent(type="text", text="3")], structuredContent={"sum": 3}),
    )
    adapter = make_adapter(client)

    [tool] = await adapter.get_tools()

    assert tool.name == "add"
    assert tool.args_schema == {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
    }
    assert tool.metadata == {"mcp": {"title": "Add numbers"}}
    message = await tool.ainvoke(
        {"name": "add", "args": {"a": 1, "b": 2}, "id": "call-1", "type": "tool_call"}
    )
    assert message.content == [{"type": "text", "text": "3", "annotations": None, "_meta": None}]
    assert message.artifact == {"mcp": {"structured_content": {"sum": 3}}}
    assert client.calls == [("add", {"a": 1, "b": 2})]


@pytest.mark.asyncio
async def test_tools_work_directly_with_create_agent() -> None:
    client = FakeClient(
        [Tool(name="add", description="Add numbers.", inputSchema={})],
        CallToolResult(content=[TextContent(type="text", text="3")]),
    )
    agent = create_agent(
        FakeToolCallingModel(
            tool_calls=[[{"name": "add", "args": {"a": 1, "b": 2}, "id": "call-1"}], []]
        ),
        await make_adapter(client).get_tools(),
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "Add 1 and 2."}]})

    assert result["messages"][-1].content.endswith("-3")
    assert client.calls == [("add", {"a": 1, "b": 2})]


@pytest.mark.asyncio
async def test_get_tools_rejects_duplicate_names() -> None:
    client = FakeClient(
        [Tool(name="duplicate", inputSchema={}), Tool(name="duplicate", inputSchema={})],
        CallToolResult(content=[]),
    )

    with pytest.raises(ValueError, match="duplicate tool names: duplicate"):
        await make_adapter(client).get_tools()


@pytest.mark.asyncio
async def test_url_target_can_be_constructed_from_an_async_context() -> None:
    MCPAdapter("https://example.com/mcp")


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


@pytest.mark.asyncio
async def test_closed_adapter_rejects_further_use() -> None:
    adapter = make_adapter(FakeClient([], CallToolResult(content=[])))
    await adapter.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await adapter.get_tools()
