"""Servers from two MCP protocol eras, in one agent.

MCP changed how a client and server agree on what each supports: the 2025-11-25
era negotiates with an `initialize` handshake, the 2026-07-28 era with
`server/discover` instead. FastMCP negotiates per connection, so tools from both
can end up in the same agent and nothing on this side has to know which is
which.

Note the shape: one adapter per server, not one `MCPConfig` fleet naming both.
A fleet is composed behind a single client, and that composite negotiates one
era for everything it holds. A fleet of modern servers does negotiate the modern
era — but add a backend that only speaks the handshake era and the whole fleet
drops to it. Separate adapters are what let each connection keep the best era
its own server supports.

    uv run examples/mcp/protocol_eras.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from _servers import calculator_server, weather_server
from fastmcp.client import Client

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter


async def main() -> None:
    """Discover tools over each era, then hand them all to one agent."""
    # `mode` picks the era. "legacy" pins the handshake; "auto" negotiates the
    # newest the server understands. Each client negotiates independently.
    legacy: Client[Any] = Client(weather_server(), mode="legacy")
    modern: Client[Any] = Client(calculator_server(), mode="auto")

    async with MCPAdapter(legacy) as legacy_adapter, MCPAdapter(modern) as modern_adapter:
        tools = await legacy_adapter.list_tools() + await modern_adapter.list_tools()

        # Only the handshake era populates `initialize_result`, so it doubles as
        # proof of which negotiation actually ran.
        print(
            f"legacy server  -> {legacy.protocol_version} (handshake ran: "
            f"{legacy.initialize_result is not None})"
        )
        print(
            f"modern server  -> {modern.protocol_version} (handshake ran: "
            f"{modern.initialize_result is not None})"
        )

    print("tools:", [tool.name for tool in tools])

    agent = create_agent("anthropic:claude-sonnet-5", tools)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Forecast for Oslo? And what is 84 / 4?"}]}
    )
    print("\nagent:", result["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
