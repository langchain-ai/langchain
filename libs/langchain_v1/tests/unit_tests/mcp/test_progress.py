"""Progress delivery through real in-memory MCP sessions."""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Literal, TypedDict

import pytest
from fastmcp import Client, Context, FastMCP
from fastmcp.client.group import ClientGroup
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from langchain.mcp import MCPAdapter


@pytest.mark.parametrize("mode", ["legacy", "auto"])
@pytest.mark.parametrize("grouped", [False, True])
async def test_adapted_tools_preserve_configured_progress_handler(
    mode: Literal["auto", "legacy"], *, grouped: bool
) -> None:
    server: FastMCP[None] = FastMCP("worker")
    events: list[tuple[float, float | None, str | None]] = []

    @server.tool
    async def work(ctx: Context) -> str:
        """Report progress before returning."""
        await ctx.report_progress(0, None, None)
        await ctx.report_progress(1, 2, "working")
        await ctx.report_progress(2, 2, "done")
        return "complete"

    async def record(progress: float, total: float | None, message: str | None) -> None:
        events.append((progress, total, message))

    client = Client(server, mode=mode, progress_handler=record)
    target = ClientGroup({"worker": client}) if grouped else client
    async with MCPAdapter(target) as adapter:
        adapted_clients = (
            list(adapter.client.clients.values())
            if isinstance(adapter.client, ClientGroup)
            else [adapter.client]
        )
        [tool] = await adapter.list_tools()
        [result] = await tool.ainvoke({})

    assert result["text"] == "complete"
    assert events == [(0, None, None), (1, 2, "working"), (2, 2, "done")]
    assert not client.is_connected()
    assert all(not member.is_connected() for member in adapted_clients)


async def test_group_progress_reaches_the_serving_members_handler() -> None:
    events: dict[str, list[str | None]] = {"first": [], "second": []}

    def member(name: str) -> Client[Any]:
        server: FastMCP[None] = FastMCP(name)

        @server.tool
        async def work(ctx: Context) -> str:
            """Report this worker's identity."""
            await ctx.report_progress(1, 1, name)
            return name

        async def record(_progress: float, _total: float | None, message: str | None) -> None:
            events[name].append(message)

        return Client(server, progress_handler=record)

    group = ClientGroup({name: member(name) for name in events})
    async with MCPAdapter(group) as adapter:
        tools = await adapter.list_tools()
        await asyncio.gather(*(tool.ainvoke({}) for tool in tools))

    assert events == {"first": ["first"], "second": ["second"]}
    assert all(not client.is_connected() for client in group.clients.values())


class _State(TypedDict):
    result: str


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Async graph context requires Python 3.11")
async def test_graph_stream_receives_progress_before_the_tool_completes() -> None:
    server: FastMCP[None] = FastMCP("worker")
    release = asyncio.Event()
    completed = asyncio.Event()

    @server.tool
    async def work(ctx: Context) -> str:
        """Wait for the consumer to receive progress before completing."""
        await ctx.report_progress(1, 2, "working")
        await release.wait()
        completed.set()
        return "complete"

    async def node(state: _State) -> _State:
        writer = get_stream_writer()

        async def record(progress: float, total: float | None, message: str | None) -> None:
            writer({"progress": progress, "total": total, "message": message})

        async with MCPAdapter(Client(server, progress_handler=record)) as adapter:
            [tool] = await adapter.list_tools()
            [result] = await tool.ainvoke({})
        return {**state, "result": result["text"]}

    builder = StateGraph(_State)
    builder.add_node("worker", node)
    builder.add_edge(START, "worker")
    builder.add_edge("worker", END)
    graph = builder.compile()

    async def consume() -> None:
        async for mode, data in graph.astream({"result": ""}, stream_mode=["custom", "updates"]):
            if mode == "custom":
                assert data == {"progress": 1, "total": 2, "message": "working"}
                assert not completed.is_set()
                release.set()
            else:
                assert data == {"worker": {"result": "complete"}}

    await asyncio.wait_for(consume(), timeout=5)
    assert release.is_set()
    assert completed.is_set()
