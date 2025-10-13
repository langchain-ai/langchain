"""Unit tests for PlanningMiddleware."""

from __future__ import annotations

from typing import cast

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain.agents.middleware.planning import PlanningMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langgraph.runtime import Runtime


def _fake_runtime() -> Runtime:
    return cast(Runtime, object())


def _make_request(system_prompt: str | None = None) -> ModelRequest:
    """Create a minimal ModelRequest for testing."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    return ModelRequest(
        model=model,
        system_prompt=system_prompt,
        messages=[],
        tool_choice=None,
        tools=[],
        response_format=None,
        state=cast("AgentState", {}),  # type: ignore[name-defined]
        runtime=_fake_runtime(),
        model_settings={},
    )


def test_adds_system_prompt_when_none_exists() -> None:
    """Test that middleware adds system prompt when request has none."""
    middleware = PlanningMiddleware()
    request = _make_request(system_prompt=None)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # System prompt should be set
    assert request.system_prompt is not None
    assert "write_todos" in request.system_prompt


def test_appends_to_existing_system_prompt() -> None:
    """Test that middleware appends to existing system prompt."""
    existing_prompt = "You are a helpful assistant."
    middleware = PlanningMiddleware()
    request = _make_request(system_prompt=existing_prompt)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # System prompt should contain both
    assert request.system_prompt is not None
    assert existing_prompt in request.system_prompt
    assert "write_todos" in request.system_prompt
    assert request.system_prompt.startswith(existing_prompt)


def test_custom_system_prompt() -> None:
    """Test that middleware uses custom system prompt."""
    custom_prompt = "Custom planning instructions"
    middleware = PlanningMiddleware(system_prompt=custom_prompt)
    request = _make_request(system_prompt=None)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # Should use custom prompt
    assert request.system_prompt == custom_prompt


def test_has_write_todos_tool() -> None:
    """Test that middleware registers the write_todos tool."""
    middleware = PlanningMiddleware()

    # Should have one tool registered
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "write_todos"


def test_custom_tool_description() -> None:
    """Test that middleware uses custom tool description."""
    custom_description = "Custom todo tool description"
    middleware = PlanningMiddleware(tool_description=custom_description)

    # Tool should use custom description
    assert len(middleware.tools) == 1
    assert middleware.tools[0].description == custom_description


# ==============================================================================
# Async Tests
# ==============================================================================


async def test_adds_system_prompt_when_none_exists_async() -> None:
    """Test async version - middleware adds system prompt when request has none."""
    middleware = PlanningMiddleware()
    request = _make_request(system_prompt=None)

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    # System prompt should be set
    assert request.system_prompt is not None
    assert "write_todos" in request.system_prompt


async def test_appends_to_existing_system_prompt_async() -> None:
    """Test async version - middleware appends to existing system prompt."""
    existing_prompt = "You are a helpful assistant."
    middleware = PlanningMiddleware()
    request = _make_request(system_prompt=existing_prompt)

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    # System prompt should contain both
    assert request.system_prompt is not None
    assert existing_prompt in request.system_prompt
    assert "write_todos" in request.system_prompt
    assert request.system_prompt.startswith(existing_prompt)


async def test_custom_system_prompt_async() -> None:
    """Test async version - middleware uses custom system prompt."""
    custom_prompt = "Custom planning instructions"
    middleware = PlanningMiddleware(system_prompt=custom_prompt)
    request = _make_request(system_prompt=None)

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    # Should use custom prompt
    assert request.system_prompt == custom_prompt


async def test_handler_called_with_modified_request_async() -> None:
    """Test async version - handler receives the modified request."""
    middleware = PlanningMiddleware()
    request = _make_request(system_prompt="Original")
    handler_called = {"value": False}
    received_prompt = {"value": None}

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        handler_called["value"] = True
        received_prompt["value"] = req.system_prompt
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    assert handler_called["value"]
    assert received_prompt["value"] is not None
    assert "Original" in received_prompt["value"]
    assert "write_todos" in received_prompt["value"]
