"""Multi-server `MCPAdapter` coverage across MCP protocol eras.

These tests need servers reachable over HTTP, because a multi-server `MCPConfig`
addresses its backends by URL. They bind loopback ports only.

Two eras are exercised:

- the handshake era, negotiated with `initialize`, served here over SSE (whose
  client transport is handshake-only)
- the modern era, negotiated with `server/discover`, served over streamable HTTP

The modern server uses `json_response=True`. The default SSE response path has a
race in the SDK's handshake-era POST handling that drops the `initialize`
response, so JSON responses keep these tests deterministic.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, cast

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.transports.config import MCPConfigTransport
from fastmcp.utilities.tests import run_server_in_process

from langchain.mcp import MCPAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator

_HANDSHAKE_ERA = "2025-11-25"
"""Protocol version that negotiates with the `initialize` handshake."""

_MODERN_ERA = "2026-07-28"
"""Protocol version that negotiates with `server/discover` instead."""


def _self_identifying_server(name: str) -> FastMCP[None]:
    """Build a server whose only tool reports which server answered.

    Args:
        name: Server name, also the value its tool returns.
    """
    server: FastMCP[None] = FastMCP(name)

    @server.tool
    def whoami() -> str:
        """Report the name of the server that handled this call."""
        return name

    return server


def _run_handshake_server(host: str, port: int) -> None:
    """Serve the handshake-era server over SSE, whose transport is era-locked."""
    _self_identifying_server("handshake-server").run(
        transport="sse", host=host, port=port, show_banner=False
    )


def _run_modern_server(host: str, port: int) -> None:
    """Serve the modern-era server over streamable HTTP.

    `json_response=True` avoids a race in the SDK's handshake-era SSE POST
    handling that can drop the `initialize` response.
    """
    _self_identifying_server("modern-server").run(
        transport="http", host=host, port=port, json_response=True, show_banner=False
    )


@contextlib.contextmanager
def _mixed_era_fleet() -> Iterator[dict[str, str]]:
    """Serve one handshake-era and one modern-era server.

    Yields:
        Mapping of era label to that server's URL.
    """
    with (
        run_server_in_process(_run_handshake_server) as handshake_url,
        run_server_in_process(_run_modern_server) as modern_url,
    ):
        yield {"handshake": f"{handshake_url}/sse", "modern": f"{modern_url}/mcp"}


@pytest.mark.asyncio
async def test_one_adapter_serves_a_fleet_spanning_both_protocol_eras() -> None:
    """A single adapter reaches backends on different protocol eras.

    Both backends expose the identically named `whoami` tool. FastMCP prefixes
    each backend's tools with its config key, so the two stay addressable
    through one adapter instead of colliding.
    """
    with _mixed_era_fleet() as urls:
        config = {
            "mcpServers": {
                "handshake": {"url": urls["handshake"], "transport": "sse"},
                "modern": {"url": urls["modern"]},
            }
        }

        async with MCPAdapter(config) as adapter:
            tools = await adapter.list_tools()
            assert sorted(tool.name for tool in tools) == ["handshake_whoami", "modern_whoami"]

            answers = {
                tool.name: (
                    await tool.ainvoke(
                        {"name": tool.name, "args": {}, "id": "c1", "type": "tool_call"}
                    )
                ).content[0]["text"]
                for tool in tools
            }
            assert answers == {
                "handshake_whoami": "handshake-server",
                "modern_whoami": "modern-server",
            }


@pytest.mark.asyncio
async def test_a_handshake_era_backend_pulls_the_whole_fleet_onto_its_era() -> None:
    """A composite exposes one era, so its oldest backend decides which.

    Documents why the fleet above still works: FastMCP reconnects the modern
    backend under the handshake era rather than serving two eras at once.
    """
    with _mixed_era_fleet() as urls:
        transport = MCPConfigTransport(
            {
                "mcpServers": {
                    "handshake": {"url": urls["handshake"], "transport": "sse"},
                    "modern": {"url": urls["modern"]},
                }
            }
        )

        async with MCPAdapter(Client(transport)) as adapter:
            await adapter.list_tools()

            assert transport.legacy_only is True
            assert cast("Client[Any]", adapter.client).protocol_version == _HANDSHAKE_ERA


@pytest.mark.asyncio
async def test_separate_adapters_keep_their_own_protocol_era() -> None:
    """Per-server adapters negotiate independently over real HTTP transports."""
    with _mixed_era_fleet() as urls:
        handshake_client: Client[Any] = Client(urls["handshake"])
        modern_client: Client[Any] = Client(urls["modern"])

        async with (
            MCPAdapter(handshake_client) as handshake_adapter,
            MCPAdapter(modern_client) as modern_adapter,
        ):
            [handshake_tool] = await handshake_adapter.list_tools()
            [modern_tool] = await modern_adapter.list_tools()

            call = {"name": "whoami", "args": {}, "id": "c1", "type": "tool_call"}
            answers = await asyncio.gather(handshake_tool.ainvoke(call), modern_tool.ainvoke(call))

            assert [message.content[0]["text"] for message in answers] == [
                "handshake-server",
                "modern-server",
            ]
            assert handshake_client.protocol_version == _HANDSHAKE_ERA
            assert modern_client.protocol_version == _MODERN_ERA
            assert handshake_client.initialize_result is not None
            assert modern_client.initialize_result is None
