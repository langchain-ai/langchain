from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from pathlib import Path

    from langchain.mcp import MCPElicitation, MCPElicitationResponse, MCPElicitationResume

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import CallToolResult, TextContent, Tool

from langchain import mcp
from langchain.agents import create_agent
from langchain.mcp.adapter import MCPAdapter
from langchain.mcp.tools import _to_fastmcp_result
from tests.unit_tests.agents.model import FakeToolCallingModel


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
    adapter = MCPAdapter(StreamableHttpTransport("https://example.com/mcp"))
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
    assert await tool.coroutine(a=1, b=2) == (
        [{"type": "text", "text": "3", "annotations": None, "_meta": None}],
        {"mcp": {"structured_content": {"sum": 3}}},
    )
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
async def test_get_tools_caches_discovery_until_refreshed() -> None:
    remote_tool = Tool(name="add", inputSchema={})
    client = FakeClient([remote_tool], CallToolResult(content=[TextContent(type="text", text="3")]))
    adapter = make_adapter(client)

    first = await adapter.get_tools()
    second = await adapter.get_tools()
    refreshed = await adapter.get_tools(refresh=True)

    assert first is second
    assert refreshed is not first
    assert client.enter_count == 2
    assert client.exit_count == 2


@pytest.mark.asyncio
async def test_url_target_can_be_constructed_from_an_async_context() -> None:
    MCPAdapter("https://example.com/mcp")


@pytest.mark.asyncio
async def test_get_tools_rejects_duplicate_names() -> None:
    client = FakeClient(
        [Tool(name="duplicate", inputSchema={}), Tool(name="duplicate", inputSchema={})],
        CallToolResult(content=[]),
    )

    with pytest.raises(ValueError, match="duplicate tool names: duplicate"):
        await make_adapter(client).get_tools()


def test_elicitation_types_are_exported_from_langchain_mcp() -> None:
    assert mcp.MCPElicitation is not None
    assert mcp.MCPElicitationResponse is not None
    assert mcp.MCPElicitationResume is not None

    request: MCPElicitation = {
        "type": "mcp_elicitation",
        "mode": "url",
        "server": "weather",
        "message": "Sign in to continue.",
        "url": "https://example.com/authorize",
    }
    response: MCPElicitationResponse = {"action": "decline"}
    resume: MCPElicitationResume = {"request-1": response}

    assert request["server"] == "weather"
    assert isinstance(resume, dict)
    assert resume["request-1"]["action"] == "decline"


def test_elicitation_resume_is_converted_for_fastmcp() -> None:
    assert _to_fastmcp_result({"action": "accept", "content": {"approved": True}}, mode="form") == {
        "action": "accept",
        "content": {"approved": True},
    }
    assert _to_fastmcp_result({"action": "decline"}, mode="url") == {"action": "decline"}


def test_stdio_target_requires_explicit_opt_in(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    script.touch()

    with pytest.raises(ValueError, match="allow_stdio=True"):
        MCPAdapter(script)

    MCPAdapter(script, allow_stdio=True)
