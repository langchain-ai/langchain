"""One long-lived adapter, shared by every run of a `langgraph dev` graph.

A graph factory can be called for every run, so an `MCPAdapter` built inside it
is built per run too — and reconnecting is not cheap. A multi-server config
does not hold one connection: FastMCP mounts one proxy per backend, each with
its own `httpx` client, and rebuilds that whole set every time the session is
opened. An adapter that is never held open therefore reconnects the entire
fleet on every tool call, not just on every run.

So the adapter is built once at module scope and its session is opened on the
first run and never closed. The factory only reads the tools that session
already discovered. Tools hold the client rather than the adapter, so they stay
callable for as long as the process does.

Never closing it is deliberate, and costs nothing at exit: the session runs in
a background task, so tearing down the event loop unwinds it and closes every
backend client. What it does not survive is a *reload* — re-importing this
module builds a second adapter while the first one's session is still open, so
a host that hot-reloads leaks a session per reload. If your host offers a
lifespan hook, close the adapter there instead of relying on process exit.

Connections are shared across runs, but not across servers: each backend gets
its own `httpx` client, and one client cannot serve several backends, since
FastMCP opens the client it is handed and httpx refuses a second open. If a
fleet lives behind a single origin, give the config one entry pointing at that
gateway rather than one entry per server — a single-server config skips the
router entirely and puts every request through one pool. That is one pool, not
one socket: concurrent calls still open parallel connections to the origin.

To tune a backend's HTTP client (limits, proxies, TLS) rather than share it,
give the config a server that builds its own transport:

    class TunedServer(RemoteMCPServer):
        def to_transport(self) -> StreamableHttpTransport:
            return StreamableHttpTransport(self.url, httpx_client_factory=my_factory)

    CONFIG = MCPConfig(mcpServers={"weather": TunedServer(url=...)})

    uv run examples/mcp/graph_factory.py

To serve it with `langgraph dev`, start the servers yourself, point
`MCP_WEATHER_URL` and `MCP_CALCULATOR_URL` at them, and name the factory in a
`langgraph.json`:

    {"dependencies": ["."], "graphs": {"fleet": "./examples/mcp/graph_factory.py:make_graph"}}
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from _servers import run_calculator_http, run_weather_http
from fastmcp.utilities.tests import run_server_in_process

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

WEATHER_PORT = 8931
CALCULATOR_PORT = 8932

CONFIG = {
    "mcpServers": {
        "weather": {
            "url": os.environ.get("MCP_WEATHER_URL", f"http://127.0.0.1:{WEATHER_PORT}/mcp")
        },
        "calc": {
            "url": os.environ.get("MCP_CALCULATOR_URL", f"http://127.0.0.1:{CALCULATOR_PORT}/mcp")
        },
    }
}

SYSTEM_PROMPT = "You answer questions using the weather and calculator tools available to you."

# Module scope on purpose: constructing an adapter connects nothing, so this is
# cheap at import time and safe before an event loop exists.
_ADAPTER = MCPAdapter(CONFIG)
_TOOLS: list[BaseTool] = []


async def fleet_tools() -> list[BaseTool]:
    """Return the fleet's tools, connecting on the first call only."""
    if not _TOOLS:
        # Opened and never closed, so every later run reuses this one session.
        await _ADAPTER.__aenter__()
        # Assigning into the list rather than extending it keeps two runs that
        # race the first call from discovering the same tools twice.
        _TOOLS[:] = await _ADAPTER.get_tools()
    return _TOOLS


async def make_graph(config: dict[str, Any] | None = None) -> CompiledStateGraph:  # noqa: ARG001
    """Build the agent for one run, over the already-connected fleet.

    The signature is what `langgraph dev` calls; `config` is unused here, but a
    factory that varies tools per assistant would read it.
    """
    return create_agent(
        "anthropic:claude-sonnet-5",
        await fleet_tools(),
        system_prompt=SYSTEM_PROMPT,
    )


async def main() -> None:
    """Build the graph twice, and show the second build reusing the first's session."""
    await make_graph()
    print("run 1:", [tool.name for tool in _TOOLS], "connected:", _ADAPTER.client.is_connected())

    await make_graph()
    print("run 2: reused the same tools:", await fleet_tools() is _TOOLS)


if __name__ == "__main__":
    # These two stand in for a real fleet. Under `langgraph dev` the servers are
    # already running and reached through the environment variables above.
    with (
        run_server_in_process(run_weather_http, port=WEATHER_PORT),
        run_server_in_process(run_calculator_http, port=CALCULATOR_PORT),
    ):
        asyncio.run(main())
