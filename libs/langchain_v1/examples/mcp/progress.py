"""Stream MCP tool progress through LangGraph without a model or network access.

uv run --extra mcp examples/mcp/progress.py
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from fastmcp import Client, Context, FastMCP
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from langchain.mcp import MCPAdapter


class State(TypedDict):
    """The worker's completed result."""

    result: str


server: FastMCP[None] = FastMCP("worker")


@server.tool
async def process_documents(ctx: Context) -> str:
    """Process a small batch and report each completed document."""
    for completed in range(1, 4):
        await ctx.report_progress(completed, 3, f"Processed document {completed}")
    return "Processed 3 documents"


async def run_worker(state: State) -> State:
    """Capture this run's writer before entering the MCP session.

    Args:
        state: Current graph state.
    """
    writer = get_stream_writer()

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        writer(
            {
                "type": "mcp_progress",
                "server": "worker",
                "tool": "process_documents",
                "progress": progress,
                "total": total,
                "message": message,
            }
        )

    client = Client(server, progress_handler=on_progress)
    async with MCPAdapter(client) as adapter:
        [tool] = await adapter.list_tools()
        [block] = await tool.ainvoke({})
    return {**state, "result": block["text"]}


async def main() -> None:
    """Print progress events followed by the completed graph state."""
    builder = StateGraph(State)
    builder.add_node("worker", run_worker)
    builder.add_edge(START, "worker")
    builder.add_edge("worker", END)
    graph = builder.compile()

    async for mode, data in graph.astream({"result": ""}, stream_mode=["custom", "updates"]):
        print(mode, data)


if __name__ == "__main__":
    asyncio.run(main())
