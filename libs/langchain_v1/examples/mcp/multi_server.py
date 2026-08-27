"""Several MCP servers behind one adapter.

An `MCPConfig` dict names each backend. FastMCP connects to all of them and
prefixes every tool with its config key, so two servers exposing the same tool
name stay distinguishable in the list handed to a model.

    uv run examples/mcp/multi_server.py
"""

from __future__ import annotations

import asyncio

from _servers import calculator_server, free_port, serve_http, weather_server

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter


async def main() -> None:
    """Give one agent the tools of two servers reached over different transports."""
    weather_port, calc_port = free_port(), free_port()
    with (
        serve_http(weather_server(), weather_port) as weather_url,
        serve_http(calculator_server(), calc_port) as calc_url,
    ):
        config = {
            "mcpServers": {
                "weather": {"url": weather_url},
                "calc": {"url": calc_url},
            }
        }

        async with MCPAdapter(config) as adapter:
            tools = await adapter.get_tools()

        print("tools:", [tool.name for tool in tools])

        agent = create_agent("anthropic:claude-sonnet-5", tools)
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Forecast for Oslo? Also, what is 84 / 4?"}]}
        )
        print("\nagent:", result["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
