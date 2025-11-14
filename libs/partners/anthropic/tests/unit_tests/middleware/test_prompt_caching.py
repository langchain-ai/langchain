"""Tests for Anthropic prompt caching middleware."""

import warnings
from typing import Any, cast
from unittest.mock import MagicMock

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

from langchain_anthropic.chat_models import ChatAnthropic
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

    # Create a mock ChatAnthropic instance
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    fake_request = ModelRequest(
        model=mock_chat_anthropic,
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

    # Create a mock ChatAnthropic instance
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    fake_request = ModelRequest(
        model=mock_chat_anthropic,
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


async def test_anthropic_prompt_caching_middleware_async_with_system_prompt() -> None:
    """Test async path counts system prompt in message count."""
    middleware = AnthropicPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=3
    )

    # Create a mock ChatAnthropic instance
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    # Test with system prompt: 2 messages + 1 system = 3 total (meets minimum)
    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Hello"), HumanMessage("World")],
        system_prompt="You are a helpful assistant",
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello"), HumanMessage("World")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should be added when system prompt pushes count to minimum
    assert fake_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "1h"}
    }


async def test_anthropic_prompt_caching_middleware_async_default_values() -> None:
    """Test async path with default middleware initialization."""
    # Test with default values (min_messages_to_cache=0)
    middleware = AnthropicPromptCachingMiddleware()

    # Create a mock ChatAnthropic instance
    mock_chat_anthropic = MagicMock(spec=ChatAnthropic)

    # Single message should trigger caching with default settings
    fake_request = ModelRequest(
        model=mock_chat_anthropic,
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Check that model_settings were added with default values
    assert fake_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "5m"}
    }


def test_remove_cache_control_when_present() -> None:
    """Test that _remove_cache_control removes cache_control from model_settings."""
    middleware = AnthropicPromptCachingMiddleware()

    fake_request = ModelRequest(
        model=MagicMock(spec=ChatAnthropic),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={"cache_control": {"type": "ephemeral", "ttl": "5m"}},
    )

    assert "cache_control" in fake_request.model_settings
    middleware._remove_cache_control(fake_request)
    assert "cache_control" not in fake_request.model_settings


def test_remove_cache_control_safe_when_absent() -> None:
    """Test that _remove_cache_control is safe when cache_control is not present."""
    middleware = AnthropicPromptCachingMiddleware()

    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},  # Empty, no cache_control
    )

    # Should not raise an error
    middleware._remove_cache_control(fake_request)
    assert "cache_control" not in fake_request.model_settings


def test_wrap_model_call_cleans_up_for_non_anthropic_model() -> None:
    """Test cache_control removal on fallback to non-Anthropic model."""
    middleware = AnthropicPromptCachingMiddleware()

    # Simulate post-fallback state: non-Anthropic model with cache_control present
    fake_request = ModelRequest(
        model=FakeToolCallingModel(),  # Non-Anthropic model
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={
            "cache_control": {"type": "ephemeral", "ttl": "5m"}
        },  # Present from earlier
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        # Verify cache_control was removed before handler is called
        assert "cache_control" not in req.model_settings
        return ModelResponse(result=[AIMessage(content="response")])

    response = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(response, ModelResponse)
    assert "cache_control" not in fake_request.model_settings


def test_wrap_model_call_recovers_from_cache_control_type_error() -> None:
    """Test that middleware recovers from cache_control TypeError and retries."""
    middleware = AnthropicPromptCachingMiddleware()

    fake_request = ModelRequest(
        model=MagicMock(spec=ChatAnthropic),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    call_count = 0
    mock_response = ModelResponse(result=[AIMessage(content="response")])

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: simulate cache_control error
            msg = (
                "Completions.create() got an unexpected keyword argument "
                "'cache_control'"
            )
            raise TypeError(msg)
        # Second call: succeed
        return mock_response

    response = middleware.wrap_model_call(fake_request, mock_handler)

    # Verify handler was called twice (original + retry)
    assert call_count == 2
    # Verify cache_control was removed on retry
    assert "cache_control" not in fake_request.model_settings
    assert response == mock_response


def test_wrap_model_call_reraises_non_cache_control_type_error() -> None:
    """Test that non-cache_control TypeErrors are re-raised."""
    middleware = AnthropicPromptCachingMiddleware()

    fake_request = ModelRequest(
        model=MagicMock(spec=ChatAnthropic),
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
        msg = "Some other type error"
        raise TypeError(msg)

    # Should re-raise the error
    with pytest.raises(TypeError, match="Some other type error"):
        middleware.wrap_model_call(fake_request, mock_handler)


async def test_awrap_model_call_cleans_up_for_non_anthropic_model() -> None:
    """Test that async version also cleans up cache_control."""
    middleware = AnthropicPromptCachingMiddleware()

    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={"cache_control": {"type": "ephemeral", "ttl": "5m"}},
    )

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        # Verify cache_control was removed before handler is called
        assert "cache_control" not in req.model_settings
        return ModelResponse(result=[AIMessage(content="response")])

    response = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(response, ModelResponse)
    assert "cache_control" not in fake_request.model_settings


async def test_awrap_model_call_recovers_from_type_error() -> None:
    """Test that async version recovers from cache_control TypeError."""
    middleware = AnthropicPromptCachingMiddleware()

    fake_request = ModelRequest(
        model=MagicMock(spec=ChatAnthropic),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    call_count = 0
    mock_response = ModelResponse(result=[AIMessage(content="response")])

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "got an unexpected keyword argument 'cache_control'"
            raise TypeError(msg)
        return mock_response

    response = await middleware.awrap_model_call(fake_request, mock_handler)

    # Verify retry happened
    assert call_count == 2
    assert "cache_control" not in fake_request.model_settings
    assert response == mock_response
