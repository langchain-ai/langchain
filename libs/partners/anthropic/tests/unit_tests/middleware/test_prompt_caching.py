"""Tests for Anthropic prompt caching middleware."""

import warnings
from typing import Any, cast

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.runtime import Runtime

from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware


class FakeToolCallingModel(BaseChatModel):
    """Fake model for testing middleware."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        messages_string = "-".join([str(m.content) for m in messages])
        message = AIMessage(content=messages_string, id="0")
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async top level call"""
        messages_string = "-".join([str(m.content) for m in messages])
        message = AIMessage(content=messages_string, id="0")
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"


def test_anthropic_prompt_caching_middleware_initialization() -> None:
    """Test AnthropicPromptCachingMiddleware initialization."""
    # Test with custom values
    middleware = AnthropicPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=5
    )
    assert middleware.type == "ephemeral"
    assert middleware.ttl == "1h"
    assert middleware.min_messages_to_cache == 5

    # Test with default values
    middleware = AnthropicPromptCachingMiddleware()
    assert middleware.type == "ephemeral"
    assert middleware.ttl == "5m"
    assert middleware.min_messages_to_cache == 0

    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    middleware.wrap_model_call(fake_request, mock_handler)
    # Check that model_settings were passed through via the request
    assert fake_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "5m"}
    }


def test_anthropic_prompt_caching_middleware_unsupported_model() -> None:
    """Test AnthropicPromptCachingMiddleware with unsupported model."""
    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="raise")

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Since we're in the langchain-anthropic package, ChatAnthropic is always
    # available. Test that it raises an error for unsupported model instances
    with pytest.raises(
        ValueError,
        match=(
            "AnthropicPromptCachingMiddleware caching middleware only supports "
            "Anthropic models, not instances of"
        ),
    ):
        middleware.wrap_model_call(fake_request, mock_handler)

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="warn")

    # Test warn behavior for unsupported model instances
    with warnings.catch_warnings(record=True) as w:
        result = middleware.wrap_model_call(fake_request, mock_handler)
        assert isinstance(result, ModelResponse)
        assert len(w) == 1
        assert (
            "AnthropicPromptCachingMiddleware caching middleware only supports "
            "Anthropic models, not instances of"
        ) in str(w[-1].message)

    # Test ignore behavior
    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")
    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)


async def test_anthropic_prompt_caching_middleware_async() -> None:
    """Test AnthropicPromptCachingMiddleware async path."""
    # Test with custom values
    middleware = AnthropicPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=5
    )

    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")] * 6,
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")] * 6},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Check that model_settings were passed through via the request
    assert fake_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "1h"}
    }


async def test_anthropic_prompt_caching_middleware_async_unsupported_model() -> None:
    """Test AnthropicPromptCachingMiddleware async path with unsupported model."""
    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="raise")

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Test that it raises an error for unsupported model instances
    with pytest.raises(
        ValueError,
        match=(
            "AnthropicPromptCachingMiddleware caching middleware only supports "
            "Anthropic models, not instances of"
        ),
    ):
        await middleware.awrap_model_call(fake_request, mock_handler)

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="warn")

    # Test warn behavior for unsupported model instances
    with warnings.catch_warnings(record=True) as w:
        result = await middleware.awrap_model_call(fake_request, mock_handler)
        assert isinstance(result, ModelResponse)
        assert len(w) == 1
        assert (
            "AnthropicPromptCachingMiddleware caching middleware only supports "
            "Anthropic models, not instances of"
        ) in str(w[-1].message)

    # Test ignore behavior
    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")
    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)


async def test_anthropic_prompt_caching_middleware_async_min_messages() -> None:
    """Test async path respects min_messages_to_cache."""
    middleware = AnthropicPromptCachingMiddleware(min_messages_to_cache=5)

    # Test with fewer messages than minimum
    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")] * 3,
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")] * 3},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should NOT be added when message count is below minimum
    assert fake_request.model_settings == {}
