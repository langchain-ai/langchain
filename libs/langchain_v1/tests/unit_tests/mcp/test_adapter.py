"""Tests for `MCPAdapter`."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import ANY

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.group import ClientGroup
from fastmcp.client.transports import (
    FastMCPTransport,
    PythonStdioTransport,
    StreamableHttpTransport,
)
from fastmcp.client.transports.config import MCPConfigTransport
from langchain_core.messages import ToolMessage

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from mcp.client.caching import CacheMode


def _calculator() -> FastMCP[None]:
    server: FastMCP[None] = FastMCP("calc")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return server


@pytest.mark.asyncio
async def test_list_tools_adapts_every_tool_a_server_exposes() -> None:
    server = _calculator()

    @server.tool
    def negate(a: int) -> int:
        """Negate a number."""
        return -a

    tools = await MCPAdapter(server).list_tools()

    assert sorted(tool.name for tool in tools) == ["add", "negate"]


def _greeter() -> FastMCP[None]:
    """A second server whose tool name collides with the calculator's."""
    server: FastMCP[None] = FastMCP("greet")

    @server.tool
    def add(a: int, b: int) -> str:
        """Not arithmetic — same name, different server."""
        return f"greetings {a} and {b}"

    return server


def _group() -> ClientGroup:
    """Two in-process servers, each behind its own client."""
    return ClientGroup({"calc": Client(_calculator()), "greet": Client(_greeter())})


@pytest.mark.asyncio
async def test_group_tools_are_namespaced_per_server() -> None:
    tools = await MCPAdapter(_group()).list_tools()

    assert sorted(tool.name for tool in tools) == ["calc_add", "greet_add"]


@pytest.mark.asyncio
async def test_colliding_tool_names_reach_their_own_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group-backed tools keep their names so FastMCP routes each invocation."""
    adapter = MCPAdapter(_group())
    seen: list[str] = []
    real = adapter.client.call_tool

    async def spy(name: str, *args: Any, **kwargs: Any) -> Any:
        seen.append(name)
        return await real(name, *args, **kwargs)

    monkeypatch.setattr(adapter.client, "call_tool", spy)
    tools = {tool.name: tool for tool in await adapter.list_tools()}

    [calc] = await tools["calc_add"].ainvoke({"a": 1, "b": 2})
    [greet] = await tools["greet_add"].ainvoke({"a": 1, "b": 2})

    assert calc["text"] == "3"
    assert greet["text"] == "greetings 1 and 2"


@pytest.mark.asyncio
async def test_group_tools_stay_callable_after_the_adapter_context_exits() -> None:
    """Tools hold their member client, which reconnects on its own."""
    async with MCPAdapter(_group()) as adapter:
        tools = {tool.name: tool for tool in await adapter.list_tools()}

    [block] = await tools["calc_add"].ainvoke({"a": 2, "b": 3})

    assert block["text"] == "5"


@pytest.mark.asyncio
async def test_list_tools_inside_a_group_context_does_not_re_enter() -> None:
    """`ClientGroup` counts its own nesting, so discovery inside a context is safe."""
    adapter = MCPAdapter(_group())

    async with adapter:
        first = await adapter.list_tools()
        # A second discovery inside the same context must not raise either.
        second = await adapter.list_tools()

    assert len(first) == len(second) == 2


@pytest.mark.asyncio
async def test_list_tools_forwards_cache_mode_to_a_single_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`cache_mode` reaches the client's discovery call, so a cache is honored.

    Defaults to `use`, which is the whole point: a bare `list_tools` would
    otherwise take the client's own default and a configured cache would sit
    unused.
    """
    adapter = MCPAdapter(_calculator())
    seen: list[str] = []
    real = adapter.client.list_tools

    async def spy(*args: Any, cache_mode: str = "use", **kwargs: Any) -> Any:
        seen.append(cache_mode)
        return await real(*args, cache_mode=cast("CacheMode", cache_mode), **kwargs)

    monkeypatch.setattr(adapter.client, "list_tools", spy)

    await adapter.list_tools()
    await adapter.list_tools(cache_mode="refresh")

    assert seen == ["use", "refresh"]


@pytest.mark.asyncio
async def test_list_tools_forwards_cache_mode_to_a_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group hardcodes `refresh` on a bare call, so the adapter must pass it.

    Without this the group's own default (`refresh`) would win and per-user
    caches keyed on the client would never be served.
    """
    adapter = MCPAdapter(_group())
    seen: list[str] = []
    real = adapter.client.list_tools

    async def spy(*args: Any, cache_mode: str = "refresh", **kwargs: Any) -> Any:
        seen.append(cache_mode)
        return await real(*args, cache_mode=cast("CacheMode", cache_mode), **kwargs)

    monkeypatch.setattr(adapter.client, "list_tools", spy)

    await adapter.list_tools()
    await adapter.list_tools(cache_mode="bypass")

    assert seen == ["use", "bypass"]


@pytest.mark.asyncio
async def test_arming_a_group_leaves_the_callers_clients_untouched() -> None:
    """Arming clones keeps a caller's own clients as they built them."""
    group = _group()
    adapter = MCPAdapter(group)

    assert adapter.client is not group
    adapter_group = cast("ClientGroup", adapter.client)
    assert set(adapter_group.clients) == set(group.clients)
    for name, member in group.clients.items():
        assert adapter_group.clients[name] is not member


@pytest.mark.asyncio
async def test_tools_work_directly_with_create_agent() -> None:
    agent = create_agent(
        FakeToolCallingModel(
            tool_calls=[[{"name": "add", "args": {"a": 1, "b": 2}, "id": "call-1"}], []]
        ),
        await MCPAdapter(_calculator()).list_tools(),
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "Add 1 and 2."}]})

    assert result["messages"][-1].content.endswith("-3")


@pytest.mark.asyncio
async def test_tools_stay_callable_after_the_adapter_context_exits() -> None:
    """Discovery inside a context must not leave the returned tools dead."""
    async with MCPAdapter(_calculator()) as adapter:
        [tool] = await adapter.list_tools()

    message = await tool.ainvoke(
        {"name": "add", "args": {"a": 2, "b": 3}, "id": "c1", "type": "tool_call"}
    )

    assert message.content[0]["text"] == "5"


def test_target_inference_is_delegated_to_fastmcp(tmp_path: Path) -> None:
    """Non-string targets FastMCP understands pass through without gatekeeping."""
    script = tmp_path / "server.py"
    script.touch()

    assert isinstance(
        cast("Client[Any]", MCPAdapter(script).client).transport, PythonStdioTransport
    )
    assert isinstance(
        cast("Client[Any]", MCPAdapter(FastMCP("in-process")).client).transport,
        FastMCPTransport,
    )


def test_url_strings_are_accepted() -> None:
    """The reading a string target is meant to have still works."""
    for url in ("https://example.com/mcp", "http://localhost:2024/mcp"):
        assert isinstance(
            cast("Client[Any]", MCPAdapter(url).client).transport,
            StreamableHttpTransport,
        )


def test_a_string_naming_a_local_script_is_refused(tmp_path: Path) -> None:
    """A string must not select subprocess execution just by existing on disk."""
    script = tmp_path / "server.py"
    script.touch()

    with pytest.raises(ValueError, match="not a valid URL"):
        MCPAdapter(str(script))

    # The same server, asked for explicitly, is still reachable.
    assert isinstance(
        cast("Client[Any]", MCPAdapter(script).client).transport, PythonStdioTransport
    )


@pytest.mark.parametrize(
    "target",
    [
        "server.py",
        "./server.js",
        "/usr/local/bin/server.py",
        "example.com/mcp",
        "",
    ],
)
def test_strings_that_are_not_urls_are_refused(target: str) -> None:
    with pytest.raises(ValueError, match="not a valid URL"):
        MCPAdapter(target)


@pytest.mark.parametrize(
    ("target", "scheme"),
    [
        # `AnyUrl` reads the drive letter as a scheme and accepts this, while on
        # Windows the path resolves and FastMCP launches it — so validating as a
        # URL without pinning the scheme would still spawn a subprocess there.
        (r"C:\Users\me\server.py", "c"),
        # FastMCP cannot infer a transport from these either way. Refusing them
        # by scheme only buys an error that names the problem.
        ("file:///etc/passwd", "file"),
        ("ws://example.com/mcp", "ws"),
    ],
)
def test_strings_parsing_as_non_http_urls_are_refused(target: str, scheme: str) -> None:
    with pytest.raises(ValueError, match=f"has scheme '{scheme}'"):
        MCPAdapter(target)


@pytest.mark.skipif(sys.platform == "win32", reason="`:` is not legal in a Windows filename")
def test_a_colon_in_a_filename_does_not_smuggle_a_path_past_url_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The portable form of the drive-letter case, reachable on POSIX too.

    A filename may contain a colon, so `AnyUrl` parses `a:b.py` as scheme `a`
    and accepts it while FastMCP resolves the same string to a real file and
    launches it. Pinning the scheme is what stands between those two readings,
    so the file here genuinely exists on the path FastMCP would find.
    """
    (tmp_path / "a:b.py").touch()
    monkeypatch.chdir(tmp_path)
    assert Path("a:b.py").exists()

    with pytest.raises(ValueError, match="has scheme 'a'"):
        MCPAdapter("a:b.py")


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

    transport = cast("Client[Any]", adapter.client).transport
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
        tools = await adapter.list_tools()

    assert sorted(tool.name for tool in tools) == ["alpha_whoami", "beta_whoami"]

    by_name = {tool.name: tool for tool in tools}
    assert await by_name["alpha_whoami"].ainvoke({}) == [
        {"type": "text", "text": "alpha", "id": ANY}
    ]


def test_prebuilt_client_without_a_handler_is_armed_on_a_clone() -> None:
    """A client with no elicitation handler is armed, on a clone not the original."""
    client: Client[Any] = Client("https://example.com/mcp")

    adapted = MCPAdapter(client).client
    assert adapted is not client


def test_prebuilt_client_with_a_handler_is_used_as_is() -> None:
    """A caller's own elicitation handler is honored: the client is left alone."""

    async def own_handler(*_: Any) -> Any:
        return None

    client: Client[Any] = Client("https://example.com/mcp", elicitation_handler=own_handler)

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
        [tool] = await adapter.list_tools()
        message = await tool.ainvoke(
            {"name": "whoami", "args": {}, "id": "c1", "type": "tool_call"}
        )

        # The adapter arms a clone of a handler-less client, so assert on the
        # client it actually connected. `.new()` preserves `mode`, so the clone
        # negotiates the same era.
        armed = cast("Client[Any]", adapter.client)
        assert message.content[0]["text"] == "solo"
        assert armed.protocol_version == expected_version
        assert (armed.initialize_result is not None) is expects_handshake


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
        [handshake_tool] = await handshake_adapter.list_tools()
        [modern_tool] = await modern_adapter.list_tools()

        call = {"name": "whoami", "args": {}, "id": "c1", "type": "tool_call"}
        answers = await asyncio.gather(handshake_tool.ainvoke(call), modern_tool.ainvoke(call))

        # The adapter arms a clone of each handler-less client, so assert on the
        # clients it actually connected. Each clone keeps its own `mode`, so the
        # two negotiated eras stay independent.
        armed_handshake = cast("Client[Any]", handshake_adapter.client)
        armed_modern = cast("Client[Any]", modern_adapter.client)
        assert [message.content[0]["text"] for message in answers] == ["old", "new"]
        assert armed_handshake.protocol_version == _HANDSHAKE_ERA
        assert armed_modern.protocol_version == _MODERN_ERA
        assert armed_handshake.initialize_result is not None
        assert armed_modern.initialize_result is None


@pytest.mark.asyncio
async def test_one_group_spans_both_protocol_eras() -> None:
    """A single `ClientGroup` keeps each member on its own negotiated era.

    This is the whole reason to reach for a group over a multi-server `Client`
    config: the config presents one aggregate endpoint on a single shared era,
    while a group holds one connection per server, so a legacy and a modern
    server stay in their native eras at the same time behind one adapter.
    """
    group = ClientGroup(
        {
            "old": Client(_self_identifying_server("old"), mode="legacy"),
            "new": Client(_self_identifying_server("new"), mode="auto"),
        }
    )

    async with MCPAdapter(group) as adapter:
        tools = {tool.name: tool for tool in await adapter.list_tools()}

        answers = await asyncio.gather(
            tools["old_whoami"].ainvoke(
                {"name": "old_whoami", "args": {}, "id": "c1", "type": "tool_call"}
            ),
            tools["new_whoami"].ainvoke(
                {"name": "new_whoami", "args": {}, "id": "c2", "type": "tool_call"}
            ),
        )

        armed = cast("ClientGroup", adapter.client)
        assert [message.content[0]["text"] for message in answers] == ["old", "new"]
        assert armed.clients["old"].protocol_version == _HANDSHAKE_ERA
        assert armed.clients["new"].protocol_version == _MODERN_ERA


@pytest.mark.asyncio
async def test_tools_from_both_protocol_eras_combine_into_one_agent() -> None:
    """One agent can hold tools discovered over both protocol eras at once.

    The two servers expose different tools, so the combined list names each
    tool exactly once and the agent's choice is unambiguous.
    """
    handshake_tools = await MCPAdapter(
        Client(_self_identifying_server("old"), mode="legacy")
    ).list_tools()
    modern_tools = await MCPAdapter(Client(_arithmetic_server("new"), mode="auto")).list_tools()

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
