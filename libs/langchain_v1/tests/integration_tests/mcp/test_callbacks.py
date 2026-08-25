"""Tests for callback functionality."""

import asyncio

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.session import ServerSession

from langchain.mcp.client import MultiServerMCPClient
from tests.integration_tests.mcp.utils import run_streamable_http


def _create_callback_server():
    """Create a server with a tool for testing callbacks."""
    server = MCPServer()

    @server.tool()
    async def execute_task(task: str, ctx: Context[ServerSession, None]) -> str:
        """Execute a task with progress and logging."""
        await ctx.info(f"Starting task: {task}")
        await ctx.report_progress(progress=0.0, total=1.0)
        await asyncio.sleep(0.01)

        await ctx.debug("Executing task...")
        await ctx.report_progress(progress=0.5, total=1.0)
        await asyncio.sleep(0.01)

        await ctx.info(f"Completed task: {task}")
        await ctx.report_progress(progress=1.0, total=1.0)
        await asyncio.sleep(0.01)

        return f"Executed: {task}"

    return server


async def test_handlers_are_passed_straight_to_the_server_connection(socket_enabled) -> None:
    """Progress and logging handlers use the MCP SDK's own signatures.

    Logging is registered per server through that connection's `session_kwargs`, so the
    handler already knows which server it belongs to and needs no injected context.
    Progress is per request, so it is passed alongside the tool call.
    """
    progress_calls = []
    logging_calls = []

    async def progress_callback(progress, total, message) -> None:
        progress_calls.append((progress, message))

    async def logging_callback(params) -> None:
        logging_calls.append(params.level)

    with run_streamable_http(_create_callback_server, 8186):
        client = MultiServerMCPClient(
            {
                "callback_test": {
                    "url": "http://localhost:8186/mcp",
                    "transport": "streamable_http",
                    "session_kwargs": {"logging_callback": logging_callback},
                }
            },
            progress_callback=progress_callback,
        )

        tools = await client.get_tools(server_name="callback_test")
        assert [tool.name for tool in tools] == ["execute_task"]

        result = await tools[0].ainvoke({"args": {"task": "test"}, "id": "1", "type": "tool_call"})
        assert any(
            "Executed: test" in block.get("text", "")
            for block in result.content
            if isinstance(block, dict)
        )

        await asyncio.sleep(0.05)  # let the notifications land
        assert len(progress_calls) >= 3, f"expected progress calls, got {progress_calls}"
        assert len(logging_calls) >= 2, f"expected log calls, got {logging_calls}"
        assert {0.0, 1.0} <= {call[0] for call in progress_calls}
        assert "info" in logging_calls
