"""Stdio server that records its PID on startup, to count connections in tests."""

import os
import sys

from mcp.server.mcpserver import MCPServer

with open(sys.argv[1], "a") as handle:  # noqa: PTH123
    handle.write(f"{os.getpid()}\n")

mcp = MCPServer("counting")


@mcp.tool()
def get_time() -> str:
    """Get current time"""
    return "5:20:00 PM EST"


if __name__ == "__main__":
    mcp.run()
