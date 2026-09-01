"""Several MCP servers behind one adapter.

An `MCPConfig` dict names each backend. FastMCP connects to all of them and
prefixes every tool with its config key, so two servers exposing the same tool
name stay distinguishable in the list handed to a model.

Each backend is addressed independently, so a fleet can mix transports — these
two are stdio, but either could be a URL.

    uv run examples/mcp/multi_server.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

_STDIO_SERVER = str(Path(__file__).parent / "_stdio_server.py")

CONFIG = {
    "mcpServers": {
        "weather": {"command": sys.executable, "args": [_STDIO_SERVER, "weather"]},
        "calc": {"command": sys.executable, "args": [_STDIO_SERVER, "calculator"]},
    }
}


async def main() -> None:
    """Give one agent the tools of two servers."""
    async with MCPAdapter(CONFIG) as adapter:
        tools = await adapter.list_tools()

    print("tools:", [tool.name for tool in tools])

    agent = create_agent("anthropic:claude-sonnet-5", tools)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Forecast for Oslo? Also, what is 84 / 4?"}]}
    )
    print("\nagent:", result["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
