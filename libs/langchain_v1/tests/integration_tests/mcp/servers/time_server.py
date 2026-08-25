from mcp.server.mcpserver import MCPServer

mcp = MCPServer("time")


@mcp.tool()
def get_time() -> str:
    """Get current time."""
    return "5:20:00 PM EST"


if __name__ == "__main__":
    mcp.run()
