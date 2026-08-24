"""Tests for callback functionality."""

import asyncio

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.session import ServerSession
from mcp.types import LoggingMessageNotificationParams

from langchain.mcp.callbacks import (
    CallbackContext,
    Callbacks,
)
from langchain.mcp.client import MultiServerMCPClient
from tests.integration_tests.mcp.utils import run_streamable_http


async def test_to_mcp_format_with_callbacks() -> None:
    """Test converting to MCP format with callbacks."""
    logging_calls = []
    progress_calls = []

    async def logging_callback(params: LoggingMessageNotificationParams, context: CallbackContext):
        logging_calls.append((params, context))

    async def progress_callback(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ):
        progress_calls.append((progress, total, message, context))

    callbacks = Callbacks(
        on_logging_message=logging_callback,
        on_progress=progress_callback,
    )
    context = CallbackContext(server_name="test_server", tool_name="test_tool")

    mcp_callbacks = callbacks.to_mcp_format(context=context)

    assert mcp_callbacks.logging_callback is not None
    assert mcp_callbacks.progress_callback is not None

    # Test logging callback
    params = LoggingMessageNotificationParams(level="info", data={"message": "test log"})
    await mcp_callbacks.logging_callback(params)
    assert len(logging_calls) == 1
    assert logging_calls[0][0] == params
    assert logging_calls[0][1].server_name == "test_server"

    # Test progress callback
    await mcp_callbacks.progress_callback(0.75, 1.0, "Almost done...")
    assert len(progress_calls) == 1
    assert progress_calls[0] == (0.75, 1.0, "Almost done...", context)


def _create_callback_server():
    """Create a server with a tool for testing callbacks."""
    server = MCPServer()

    @server.tool()
    async def execute_task(task: str, ctx: Context[ServerSession, None]) -> str:
        """Execute a task with progress and logging"""
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


async def test_callbacks_with_mcp_tool_execution(socket_enabled) -> None:
    """Test callbacks integration during MCP tool execution."""
    progress_calls = []
    logging_calls = []

    async def progress_callback(progress, total, message, context):
        progress_calls.append((progress, message, context.tool_name))

    async def logging_callback(params, context):
        logging_calls.append((params.level, context.tool_name))

    callbacks = Callbacks(
        on_progress=progress_callback,
        on_logging_message=logging_callback,
    )

    with run_streamable_http(_create_callback_server, 8186):
        client = MultiServerMCPClient(
            {
                "callback_test": {
                    "url": "http://localhost:8186/mcp",
                    "transport": "streamable_http",
                }
            },
            callbacks=callbacks,
        )

        tools = await client.get_tools(server_name="callback_test")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "execute_task"

        result = await tool.ainvoke({"args": {"task": "test"}, "id": "1", "type": "tool_call"})
        assert any(
            "Executed: test" in block.get("text", "")
            for block in result.content
            if isinstance(block, dict)
        )

        # Verify both progress and logging callbacks were called
        await asyncio.sleep(0.05)  # Give time for callbacks to complete
        assert len(progress_calls) >= 3, (
            f"Expected at least 3 progress calls, got {len(progress_calls)}"
        )
        assert len(logging_calls) >= 2, (
            f"Expected at least 2 logging calls, got {len(logging_calls)}"
        )

        # Verify progress values
        progress_values = [call[0] for call in progress_calls]
        assert 0.0 in progress_values
        assert 1.0 in progress_values

        # Verify log levels
        log_levels = [call[0] for call in logging_calls]
        assert "info" in log_levels
