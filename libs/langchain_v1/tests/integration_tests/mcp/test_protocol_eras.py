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
import socket
from typing import TYPE_CHECKING, Any

import pytest
import uvicorn
from fastmcp import Client, FastMCP
from fastmcp.client.transports.config import MCPConfigTransport

from langchain.mcp import MCPAdapter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.types import ASGIApp

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


def _free_port() -> int:
    """Reserve and release a loopback port, returning its number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ReadySignalingServer(uvicorn.Server):
    """A uvicorn server that announces startup through an `asyncio.Event`.

    `uvicorn.Server` only exposes a `started` flag, which a caller would have to
    poll. Setting an event in the startup hook lets the fixture await readiness.
    """

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.ready = asyncio.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        """Start serving, then signal readiness."""
        await super().startup(sockets=sockets)
        self.ready.set()


@contextlib.asynccontextmanager
async def _serving(app: ASGIApp, port: int) -> AsyncIterator[None]:
    """Serve `app` on a loopback port for the duration of the context.

    Args:
        app: ASGI application to serve.
        port: Loopback port to bind.

    Yields:
        Once the server reports itself started.

    Raises:
        RuntimeError: If the server task exits before signaling readiness.
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = _ReadySignalingServer(config)
    task = asyncio.create_task(server.serve())
    try:
        ready = asyncio.create_task(server.ready.wait())
        done, _ = await asyncio.wait({ready, task}, return_when=asyncio.FIRST_COMPLETED)
        if ready not in done:
            ready.cancel()
            await task  # re-raise the startup failure if there was one
            msg = f"server on port {port} exited before becoming ready"
            raise RuntimeError(msg)
        yield
    finally:
        server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await task


@contextlib.asynccontextmanager
async def _mixed_era_fleet() -> AsyncIterator[dict[str, str]]:
    """Serve one handshake-era and one modern-era server.

    Yields:
        Mapping of era label to that server's URL.
    """
    handshake_port, modern_port = _free_port(), _free_port()
    handshake_app = _self_identifying_server("handshake-server").http_app(transport="sse")
    modern_app = _self_identifying_server("modern-server").http_app(
        transport="http", json_response=True
    )

    async with _serving(handshake_app, handshake_port), _serving(modern_app, modern_port):
        yield {
            "handshake": f"http://127.0.0.1:{handshake_port}/sse",
            "modern": f"http://127.0.0.1:{modern_port}/mcp",
        }


@pytest.mark.asyncio
async def test_one_adapter_serves_a_fleet_spanning_both_protocol_eras() -> None:
    """A single adapter reaches backends on different protocol eras.

    Both backends expose the identically named `whoami` tool. FastMCP prefixes
    each backend's tools with its config key, so the two stay addressable
    through one adapter instead of colliding.
    """
    async with _mixed_era_fleet() as urls:
        config = {
            "mcpServers": {
                "handshake": {"url": urls["handshake"], "transport": "sse"},
                "modern": {"url": urls["modern"]},
            }
        }

        async with MCPAdapter(config) as adapter:
            tools = await adapter.get_tools()
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
    async with _mixed_era_fleet() as urls:
        transport = MCPConfigTransport(
            {
                "mcpServers": {
                    "handshake": {"url": urls["handshake"], "transport": "sse"},
                    "modern": {"url": urls["modern"]},
                }
            }
        )

        async with MCPAdapter(Client(transport)) as adapter:
            await adapter.get_tools()

            assert transport.legacy_only is True
            assert adapter.client.protocol_version == _HANDSHAKE_ERA


@pytest.mark.asyncio
async def test_separate_adapters_keep_their_own_protocol_era() -> None:
    """Per-server adapters negotiate independently over real HTTP transports."""
    async with _mixed_era_fleet() as urls:
        handshake_client: Client[Any] = Client(urls["handshake"])
        modern_client: Client[Any] = Client(urls["modern"])

        async with (
            MCPAdapter(handshake_client) as handshake_adapter,
            MCPAdapter(modern_client) as modern_adapter,
        ):
            [handshake_tool] = await handshake_adapter.get_tools()
            [modern_tool] = await modern_adapter.get_tools()

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
