from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "anthropic", reason="Anthropic SDK is required for Claude middleware tests"
)

from langchain_anthropic.middleware.bash import ClaudeBashToolMiddleware


def test_creates_bash_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ClaudeBashToolMiddleware inherits parent's 'shell' tool."""
    middleware = ClaudeBashToolMiddleware()

    # Should have exactly one tool registered (from parent)
    assert len(middleware.tools) == 1

    # Tool is named "shell" (from parent), replaced with "bash" in wrap_model_call
    shell_tool = middleware.tools[0]
    assert shell_tool.name == "shell"


def test_replaces_tool_with_claude_descriptor() -> None:
    """Test wrap_model_call replaces shell tool with Claude's bash descriptor."""
    from langchain.agents.middleware.types import ModelRequest

    middleware = ClaudeBashToolMiddleware()

    # Create a mock request with the shell tool (inherited from parent)
    shell_tool = middleware.tools[0]
    request = ModelRequest(
        model=MagicMock(),
        system_prompt=None,
        messages=[],
        tool_choice=None,
        tools=[shell_tool],
        response_format=None,
        state={"messages": []},
        runtime=MagicMock(),
    )

    # Mock handler that captures the modified request
    captured_request = None

    def handler(req: ModelRequest) -> MagicMock:
        nonlocal captured_request
        captured_request = req
        return MagicMock()

    middleware.wrap_model_call(request, handler)

    # The shell tool should be replaced with Claude's native bash descriptor
    assert captured_request is not None
    assert len(captured_request.tools) == 1
    assert captured_request.tools[0] == {
        "type": "bash_20250124",
        "name": "shell",
    }
