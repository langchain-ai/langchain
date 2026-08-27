"""One adapter, three transports.

`MCPAdapter` infers the transport from whatever you hand it, so the only thing
that changes between an in-process server, a local script over stdio, and a
remote URL is the target itself.

    uv run examples/mcp/transports.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from _servers import free_port, serve_http, weather_server

from langchain.mcp import MCPAdapter

_STDIO_SERVER = Path(__file__).parent / "_stdio_server.py"


async def show(label: str, target: object) -> None:
    """Adapt one target and call the tool it exposes."""
    async with MCPAdapter(target) as adapter:  # type: ignore[arg-type]
        [forecast] = await adapter.get_tools()
        # An MCP result arrives as LangChain content blocks, not a bare string.
        [block] = await forecast.ainvoke({"city": "Oslo"})
        print(f"{label:12} {forecast.name} -> {block['text']}")


async def main() -> None:
    """Reach the same server over each transport in turn."""
    # In-process: no subprocess, no socket. Ideal for tests.
    await show("in-memory", weather_server())

    # A script path is launched over stdio, one subprocess per adapter.
    await show("stdio", _STDIO_SERVER)

    # A URL is reached over streamable HTTP.
    with serve_http(weather_server(), free_port()) as url:
        await show("http", url)


if __name__ == "__main__":
    asyncio.run(main())
