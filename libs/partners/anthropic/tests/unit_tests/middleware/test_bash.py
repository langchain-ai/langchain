from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "anthropic", reason="Anthropic SDK is required for Claude middleware tests"
)

from langchain.agents.middleware.types import ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from langchain_anthropic.middleware.bash import ClaudeBashToolMiddleware


class _DummyModelRequest:
    def __init__(self, tools: list[object]) -> None:
        self.tools = tools

    def override(self, **kwargs: object) -> ModelRequest:
        overridden = _DummyModelRequest(list(kwargs.get("tools", self.tools)))
        # Populate required attributes expected downstream but unused in tests.
        overridden.tools = kwargs.get("tools", self.tools)
        return overridden  # type: ignore[return-value]


def test_wrap_model_call_adds_descriptor() -> None:
    middleware = ClaudeBashToolMiddleware()
    request = _DummyModelRequest([])

    def handler(updated_request: ModelRequest) -> ModelRequest:  # type: ignore[override]
        return updated_request

    result = middleware.wrap_model_call(request, handler)
    assert result.tools[-1] == {"type": "bash_20250124", "name": "bash"}

    # Ensure we do not duplicate the descriptor on subsequent calls.
    result_again = middleware.wrap_model_call(result, handler)
    assert result_again.tools.count({"type": "bash_20250124", "name": "bash"}) == 1


def test_wrap_tool_call_handles_claude_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = ClaudeBashToolMiddleware()
    sentinel = ToolMessage(content="ok", tool_call_id="call-1", name="bash")

    monkeypatch.setattr(middleware, "_run_shell_tool", MagicMock(return_value=sentinel))
    monkeypatch.setattr(
        middleware, "_ensure_resources", MagicMock(return_value=MagicMock())
    )

    tool_call = {"name": "bash", "args": {"command": "echo hi"}, "id": "call-1"}
    request = ToolCallRequest(
        tool_call=tool_call, tool=MagicMock(), state={}, runtime=None
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
    tool_call = {"name": "other", "args": {}, "id": "call-2"}
    request = ToolCallRequest(
        tool_call=tool_call, tool=MagicMock(), state={}, runtime=None
    )

    sentinel = ToolMessage(content="handled", tool_call_id="call-2", name="other")

    def handler(_: ToolCallRequest) -> ToolMessage:
        return sentinel

    result = middleware.wrap_tool_call(request, handler)
    assert result is sentinel
