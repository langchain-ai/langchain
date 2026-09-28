"""A failing MCP tool reaches the model instead of ending the run.

When a server reports `isError=True`, the adapter turns it into a `ToolMessage`
with `status="error"` carrying the server's own message. The agent reads that
and can correct itself. Transport failures still raise, because a model cannot
act on those.

    uv run examples/mcp/tool_errors.py
"""

from __future__ import annotations

import asyncio

from _servers import calculator_server

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter


async def main() -> None:
    """Ask for a division by zero and watch the agent recover."""
    async with MCPAdapter(calculator_server()) as adapter:
        tools = await adapter.list_tools()

    agent = create_agent(
        "anthropic:claude-sonnet-5",
        tools,
        # Without this the model just answers from arithmetic it already knows,
        # and the error path never runs.
        system_prompt="Always use the `divide` tool for arithmetic. Never compute it yourself.",
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is 10 divided by 0? Then try 10 / 4."}]}
    )

    for message in result["messages"]:
        if message.type == "tool":
            print(f"tool call -> status={message.status}: {message.text.strip()[:70]}")

    print("\nagent:", result["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
