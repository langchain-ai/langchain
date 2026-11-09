"""One adapter, three transports.

`MCPAdapter` infers the transport from whatever you hand it, so the only thing
that changes between an in-process server, a local script over stdio, and a
remote URL is the target itself.

    uv run examples/mcp/transports.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from _servers import run_weather_http, weather_server
from fastmcp.utilities.tests import run_server_in_process

from langchain.mcp import MCPAdapter

if TYPE_CHECKING:
    from langchain.mcp.adapter import MCPAdapterTarget

_STDIO_SERVER = Path(__file__).parent / "_stdio_server.py"


async def show(label: str, target: MCPAdapterTarget) -> None:
    """Adapt one target and call the tool it exposes."""
    async with MCPAdapter(target) as adapter:
        [forecast] = await adapter.list_tools()
        # An MCP result arrives as LangChain content blocks, not a bare string.
        [block] = await forecast.ainvoke({"city": "Oslo"})
        print(f"{label:12} {forecast.name} -> {block['text']}")


async def main() -> None:
    """Reach the same server over each transport in turn."""
    # In-process: no subprocess, no socket. Ideal for tests.
    await show("in-memory", weather_server())

    # A script path is launched over stdio, one subprocess per adapter.
    await show("stdio", _STDIO_SERVER)

    # A URL is reached over streamable HTTP. FastMCP's own test helper runs the
    # server in a subprocess and hands back its URL.
    with run_server_in_process(run_weather_http) as url:
        await show("http", f"{url}/mcp")


if __name__ == "__main__":
    asyncio.run(main())
