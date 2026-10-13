"""Gate destructive MCP tools behind human approval.

An MCP server can flag a tool as destructive (`destructiveHint=True`). The
adapter surfaces that on the LangChain tool as
`metadata["mcp"]["tool"]["annotations"]["destructive_hint"]`, so a client can
decide how to treat it without hardcoding tool names.

Here we read that hint to build the `interrupt_on` map for
`HumanInTheLoopMiddleware`: destructive tools pause for approval, everything
else runs untouched. Because the gate is derived from metadata, it covers
whatever destructive tools a server happens to expose.

    uv run examples/mcp/destructive_interrupt.py
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from _servers import files_server
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain.mcp import MCPAdapter

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def _is_destructive(tool: BaseTool) -> bool:
    """Read the MCP destructive hint off the adapter's tool metadata."""
    annotations = (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
    return annotations.get("destructive_hint", False)


def _describe(tool_call: Any, _state: Any, _runtime: Any) -> str:
    """Format the approval prompt from the pending tool call."""
    return f"Approve destructive call {tool_call['name']}({tool_call['args']})?"


async def main() -> None:
    """Run the agent so a destructive tool pauses for approval, then approve it."""
    async with MCPAdapter(files_server()) as adapter:
        tools = await adapter.list_tools()

    # The filter: build the interrupt map from metadata, not tool names.
    interrupt_on: dict[str, bool | InterruptOnConfig] = {
        tool.name: InterruptOnConfig(allowed_decisions=["approve", "reject"], description=_describe)
        for tool in tools
        if _is_destructive(tool)
    }
    print("gated tools:", sorted(interrupt_on))

    agent = create_agent(
        "anthropic:claude-sonnet-5",
        tools,
        middleware=[HumanInTheLoopMiddleware(interrupt_on=interrupt_on)],
        checkpointer=InMemorySaver(),  # resuming a paused run needs persistence
    )
    config: Any = {"configurable": {"thread_id": "files-1"}}

    paused = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Delete report.md."}]}, config
    )

    [interrupt] = paused["__interrupt__"]
    [request] = interrupt.value["action_requests"]
    print(f"paused for approval: {request['description']}")

    # Approve and let the tool run. A `reject` would skip it and tell the model.
    resumed = await agent.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    print("agent:", resumed["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
