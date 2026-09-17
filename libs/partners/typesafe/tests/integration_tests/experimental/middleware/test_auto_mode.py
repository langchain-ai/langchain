"""Live integration tests for `AutoModeMiddleware`."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents.middleware.types import AgentState, ToolCallRequest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool

from langchain_typesafe.experimental.middleware import AutoModeMiddleware


@tool
def delete_file(path: str) -> str:
    """Delete a file at the supplied path."""
    return path


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_live_classification_blocks_without_executing_tool(
    *,
    asynchronous: bool,
) -> None:
    """Block live synchronous and asynchronous tool calls deterministically."""
    middleware = AutoModeMiddleware(tools=["delete_file"], risk_threshold=0.0)
    request = ToolCallRequest(
        tool_call=ToolCall(
            name="delete_file",
            args={"path": "/workspace/report.txt"},
            id="call_live",
            type="tool_call",
        ),
        tool=delete_file,
        state=cast(
            "AgentState[Any]",
            {"messages": [HumanMessage("Summarize the report.")]},
        ),
        runtime=MagicMock(),
    )
    sync_handler = MagicMock()
    async_handler = AsyncMock()

    try:
        if asynchronous:
            result = await middleware.awrap_tool_call(request, async_handler)
        else:
            result = middleware.wrap_tool_call(request, sync_handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == "call_live"
        sync_handler.assert_not_called()
        async_handler.assert_not_awaited()
    finally:
        if middleware.classifier.async_client is not None:
            await middleware.classifier.async_client.aclose()
        if middleware.classifier.client is not None:
            middleware.classifier.client.close()
