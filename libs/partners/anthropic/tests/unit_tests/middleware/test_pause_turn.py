"""Tests for Anthropic paused-turn middleware."""

from typing import Any, cast

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from langchain_anthropic.middleware import (
    AnthropicPauseTurnMiddleware,
    PauseTurnLimitExceededError,
)


def _request() -> ModelRequest:
    return ModelRequest(
        model=cast(Any, object()),
        messages=[HumanMessage("Search for the answer")],
        tools=[],
        state={"messages": []},
        runtime=cast(Runtime, object()),
    )


def _paused() -> ModelResponse:
    return ModelResponse(
        result=[
            AIMessage(
                content=[{"type": "server_tool_use", "name": "web_search"}],
                response_metadata={"stop_reason": "pause_turn"},
            )
        ]
    )


def test_pause_turn_continues_and_preserves_messages() -> None:
    middleware = AnthropicPauseTurnMiddleware()
    requests: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        if len(requests) == 1:
            return _paused()
        return ModelResponse(
            result=[
                AIMessage(
                    content="The answer",
                    response_metadata={"stop_reason": "end_turn"},
                )
            ]
        )

    result = middleware.wrap_model_call(_request(), handler)

    assert isinstance(result, ModelResponse)
    assert len(requests) == 2
    assert len(requests[1].messages) == 2
    assert requests[1].messages[-1].response_metadata["stop_reason"] == "pause_turn"
    assert len(result.result) == 2
    assert result.result[-1].content == "The answer"


def test_pause_turn_respects_continuation_limit() -> None:
    middleware = AnthropicPauseTurnMiddleware(max_continuations=1)
    calls = 0

    def handler(_: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        return _paused()

    with pytest.raises(PauseTurnLimitExceededError, match="after 1 continuations"):
        middleware.wrap_model_call(_request(), handler)

    assert calls == 2


def test_pause_turn_without_middleware_behavior_is_not_involved() -> None:
    """A normal model response is returned without an extra model call."""
    middleware = AnthropicPauseTurnMiddleware()
    expected = ModelResponse(result=[AIMessage(content="Done")])
    calls = 0

    def handler(_: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        return expected

    result = middleware.wrap_model_call(_request(), handler)

    assert isinstance(result, ModelResponse)
    assert result.result == expected.result
    assert calls == 1


@pytest.mark.asyncio
async def test_pause_turn_async_continuation() -> None:
    middleware = AnthropicPauseTurnMiddleware()
    requests: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        if len(requests) == 1:
            return _paused()
        return ModelResponse(result=[AIMessage(content="Done")])

    result = await middleware.awrap_model_call(_request(), handler)

    assert isinstance(result, ModelResponse)
    assert len(requests) == 2
    assert len(result.result) == 2


def test_pause_turn_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        AnthropicPauseTurnMiddleware(max_continuations=-1)
