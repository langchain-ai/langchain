from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages.tool import ToolCall

pytest.importorskip(
    "anthropic", reason="Anthropic SDK is required for Claude middleware tests"
)

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage

from langchain_anthropic.middleware.bash import ClaudeBashToolMiddleware


def test_wrap_tool_call_handles_claude_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = ClaudeBashToolMiddleware()
    sentinel = ToolMessage(content="ok", tool_call_id="call-1", name="bash")

    monkeypatch.setattr(middleware, "_run_shell_tool", MagicMock(return_value=sentinel))
    monkeypatch.setattr(
        middleware, "_ensure_resources", MagicMock(return_value=MagicMock())
    )

    tool_call: ToolCall = {
        "name": "bash",
        "args": {"command": "echo hi"},
        "id": "call-1",
    }
    request = ToolCallRequest(
        tool_call=tool_call,
        tool=MagicMock(),
        state={},
        runtime=None,  # type: ignore[arg-type]
    )

    handler_called = False

    def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        return ToolMessage(content="should not be used", tool_call_id="call-1")

    result = middleware.wrap_tool_call(request, handler)
    assert result is sentinel
    assert handler_called is False


def test_wrap_tool_call_passes_through_other_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = ClaudeBashToolMiddleware()
    tool_call: ToolCall = {"name": "other", "args": {}, "id": "call-2"}
    request = ToolCallRequest(
        tool_call=tool_call,
        tool=MagicMock(),
        state={},
        runtime=None,  # type: ignore[arg-type]
    )

    sentinel = ToolMessage(content="handled", tool_call_id="call-2", name="other")

    def handler(_: ToolCallRequest) -> ToolMessage:
        return sentinel

    result = middleware.wrap_tool_call(request, handler)
    assert result is sentinel
