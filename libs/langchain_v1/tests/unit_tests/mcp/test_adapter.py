"""Tests for `MCPAdapter`."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import ANY

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.transports import (
    FastMCPTransport,
    PythonStdioTransport,
)
from fastmcp.client.transports.config import MCPConfigTransport
from langchain_core.messages import ToolMessage

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
    # FastMCP prefixes each backend's tools with its config key, so two servers
    # exposing the same tool name stay distinguishable through one adapter
    # rather than colliding in the tool list handed to a model.
    assert transport.name_as_prefix is True


_STDIO_SERVER = """
import sys
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("{name}")


@mcp.tool()
def whoami() -> str:
    \"\"\"Name the server answering this call.\"\"\"
    return "{name}"


mcp.run()
"""


@pytest.fixture
def two_stdio_servers(tmp_path: Path) -> dict[str, Any]:
    """Write two single-tool stdio servers and return a config naming both."""
    servers = {}
    for name in ("alpha", "beta"):
        script = tmp_path / f"{name}.py"
        script.write_text(_STDIO_SERVER.format(name=name))
        servers[name] = {"command": sys.executable, "args": [str(script)]}
    return {"mcpServers": servers}


@pytest.mark.asyncio
async def test_several_servers_connect_and_keep_their_prefixes(
    two_stdio_servers: dict[str, Any],
) -> None:
    """A multi-server config connects, and each backend's tools stay namespaced.

    Connecting is the point. Mounting several backends behind one client needs
    the server half of FastMCP, which a client-only install does not provide —
    a config that builds a valid transport can still fail the moment it dials.
    So this drives real servers rather than asserting on the transport.
    """
    config = two_stdio_servers

    async with MCPAdapter(config) as adapter:
        tools = await adapter.get_tools()

    assert sorted(tool.name for tool in tools) == ["alpha_whoami", "beta_whoami"]

    by_name = {tool.name: tool for tool in tools}
    assert await by_name["alpha_whoami"].ainvoke({}) == [
        {"type": "text", "text": "alpha", "id": ANY}
    ]


def test_prebuilt_client_is_used_as_is() -> None:
    client: Client[Any] = Client("https://example.com/mcp")

    assert MCPAdapter(client).client is client


_HANDSHAKE_ERA = "2025-11-25"
"""Protocol version that negotiates with the `initialize` handshake."""

_MODERN_ERA = "2026-07-28"
"""Protocol version that negotiates with `server/discover` instead."""


def _tool_text(message: ToolMessage) -> str:
    """Return the text of a `ToolMessage`'s first content block.

    Args:
        message: A `ToolMessage` produced by an adapted MCP tool.

    Returns:
        The `text` of its first content block.
    """
    blocks = message.content
    assert not isinstance(blocks, str)
    block = blocks[0]
    assert isinstance(block, dict)
    return str(block["text"])


def _self_identifying_server(name: str) -> FastMCP[None]:
    """Build a server exposing a `whoami` tool that names the answering server.

    Every server built here exposes the identical tool, so a client only ever
    reaches the server it is connected to.

    Args:
        name: Server name, also the value its tool returns.
    """
    server: FastMCP[None] = FastMCP(name)

    @server.tool
    def whoami() -> str:
        """Report the name of the server that handled this call."""
        return name

    return server


def _arithmetic_server(name: str) -> FastMCP[None]:
    """Build a server exposing a single `negate` tool.

    Args:
        name: Server name.
    """
    server: FastMCP[None] = FastMCP(name)

    @server.tool
    def negate(a: int) -> int:
        """Negate a number."""
        return -a

    return server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_version", "expects_handshake"),
    [
        ("legacy", _HANDSHAKE_ERA, True),
        ("auto", _MODERN_ERA, False),
        (_MODERN_ERA, _MODERN_ERA, False),
    ],
)
async def test_adapter_adapts_tools_on_either_protocol_era(
    mode: str,
    expected_version: str,
    expects_handshake: bool,  # noqa: FBT001
) -> None:
    """Tool discovery and calling work whichever protocol era the client negotiates.

    `initialize_result` is only populated by the handshake era, so it doubles as
    the assertion that the intended era was actually negotiated.
    """
    client: Client[Any] = Client(_self_identifying_server("solo"), mode=mode)

    async with MCPAdapter(client) as adapter:
        [tool] = await adapter.get_tools()
        message = await tool.ainvoke(
            {"name": "whoami", "args": {}, "id": "c1", "type": "tool_call"}
        )

        assert message.content[0]["text"] == "solo"
        assert client.protocol_version == expected_version
        assert (client.initialize_result is not None) is expects_handshake


@pytest.mark.asyncio
async def test_servers_on_different_protocol_eras_are_usable_side_by_side() -> None:
    """Two servers, one per protocol era, stay usable through separate adapters.

    Both servers expose the identical `whoami` tool, so this also covers the
    ordinary case of two connections whose tools have the same name: each
    client reaches only its own server. Each leg additionally has to keep its
    own negotiated era, since a shared module-level default would drag both
    adapters onto one protocol version.
    """
    handshake_client: Client[Any] = Client(_self_identifying_server("old"), mode="legacy")
    modern_client: Client[Any] = Client(_self_identifying_server("new"), mode="auto")

    async with (
        MCPAdapter(handshake_client) as handshake_adapter,
        MCPAdapter(modern_client) as modern_adapter,
    ):
        [handshake_tool] = await handshake_adapter.get_tools()
        [modern_tool] = await modern_adapter.get_tools()

        call = {"name": "whoami", "args": {}, "id": "c1", "type": "tool_call"}
        answers = await asyncio.gather(handshake_tool.ainvoke(call), modern_tool.ainvoke(call))

        assert [message.content[0]["text"] for message in answers] == ["old", "new"]
        assert handshake_client.protocol_version == _HANDSHAKE_ERA
        assert modern_client.protocol_version == _MODERN_ERA
        assert handshake_client.initialize_result is not None
        assert modern_client.initialize_result is None


@pytest.mark.asyncio
async def test_tools_from_both_protocol_eras_combine_into_one_agent() -> None:
    """One agent can hold tools discovered over both protocol eras at once.

    The two servers expose different tools, so the combined list names each
    tool exactly once and the agent's choice is unambiguous.
    """
    handshake_tools = await MCPAdapter(
        Client(_self_identifying_server("old"), mode="legacy")
    ).get_tools()
    modern_tools = await MCPAdapter(Client(_arithmetic_server("new"), mode="auto")).get_tools()

    agent = create_agent(
        FakeToolCallingModel(
            tool_calls=[
                [{"name": "whoami", "args": {}, "id": "call-1"}],
                [{"name": "negate", "args": {"a": 7}, "id": "call-2"}],
                [],
            ]
        ),
        handshake_tools + modern_tools,
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "Use both."}]})

    answered = {
        _tool_text(message) for message in result["messages"] if isinstance(message, ToolMessage)
    }
    assert answered == {"old", "-7"}
