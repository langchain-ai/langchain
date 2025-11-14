from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "anthropic", reason="Anthropic SDK is required for Claude middleware tests"
)

from langchain_anthropic.middleware.bash import ClaudeBashToolMiddleware


def test_creates_bash_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ClaudeBashToolMiddleware creates a 'bash' tool."""
    middleware = ClaudeBashToolMiddleware()

    # Should have exactly one tool registered
    assert len(middleware.tools) == 1

    # The tool should be named "bash"
    bash_tool = middleware.tools[0]
    assert bash_tool.name == "bash"


def test_replaces_tool_with_claude_descriptor() -> None:
    """Test that wrap_model_call replaces bash tool with Claude's native descriptor."""
    from langchain.agents.middleware.types import ModelRequest

    middleware = ClaudeBashToolMiddleware()

    # Create a mock request with the bash tool
    bash_tool = middleware.tools[0]
    request = ModelRequest(
        model=MagicMock(),
        system_prompt=None,
        messages=[],
        tool_choice=None,
        tools=[bash_tool],
        response_format=None,
        state={},
        runtime=MagicMock(),
    )

    # Mock handler that captures the modified request
    captured_request = None

    def handler(req: ModelRequest) -> MagicMock:
        nonlocal captured_request
        captured_request = req
        return MagicMock()

    middleware.wrap_model_call(request, handler)

    # The bash tool should be replaced with Claude's native descriptor
    assert captured_request is not None
    assert len(captured_request.tools) == 1
    assert captured_request.tools[0] == {
        "type": "bash_20250124",
        "name": "bash",
    }
