"""Minimal MCP server used by integration tests (run over stdio)."""

from fastmcp import FastMCP

server = FastMCP("langchain-mcp-test")


@server.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    server.run()
