# Handling MCP Server Connection Failures in Python

When building agentic workflows using `MultiServerMCPClient`, a temporary network drop or offline server can cause `get_tools()` to raise an exception, halting the entire application.

## Workaround / Fallback Pattern

To prevent your application from crashing when an optional MCP server is unreachable, wrap your client initialization in a `try/except` block:

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def get_safe_tools(server_configs):
    tools = []
    for name, config in server_configs.items():
        try:
            single_client = MultiServerMCPClient({name: config})
            server_tools = await single_client.get_tools()
            tools.extend(server_tools)
        except Exception as e:
            print(f"Warning: Could not connect to MCP server '{name}': {e}")
    return tools
