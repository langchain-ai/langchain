from mcp.server.fastmcp import FastMCP

mcp = FastMCP("time")


@mcp.tool()
def get_time() -> str:
    """Get current time"""
    return "5:20:00 PM EST"


if __name__ == "__main__":
    mcp.run()
