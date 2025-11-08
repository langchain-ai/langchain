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

    # Track the state during handler execution
    settings_during_call = {}

    def mock_handler(req: ModelRequest) -> ModelResponse:
        settings_during_call.update(req.model_settings)
        return ModelResponse(result=[AIMessage(content="mock response")])

    middleware.wrap_model_call(fake_request, mock_handler)
    # Check that model_settings were passed through during handler execution
    assert settings_during_call == {"cache_control": {"type": "ephemeral", "ttl": "5m"}}
    # Verify cleanup after handler completes
    assert fake_request.model_settings == {}


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

    # Track the state during handler execution
    settings_during_call = {}

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        settings_during_call.update(req.model_settings)
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Check that model_settings were passed through during handler execution
    assert settings_during_call == {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    # Verify cleanup after handler completes
    assert fake_request.model_settings == {}


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

    # Track the state during handler execution
    settings_during_call = {}

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        settings_during_call.update(req.model_settings)
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should be added when system prompt pushes count to minimum
    assert settings_during_call == {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    # Verify cleanup after handler completes
    assert fake_request.model_settings == {}


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

    # Track the state during handler execution
    settings_during_call = {}

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        settings_during_call.update(req.model_settings)
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Check that model_settings were added with default values during handler execution
    assert settings_during_call == {"cache_control": {"type": "ephemeral", "ttl": "5m"}}
    # Verify cleanup after handler completes
    assert fake_request.model_settings == {}


def test_cache_control_cleanup_on_success() -> None:
    """Test that cache_control is cleaned up after successful handler execution.

    This test verifies the fix for issue #33709 where cache_control was persisting
    in model_settings and breaking fallback middleware with non-Anthropic models.
    """
    middleware = AnthropicPromptCachingMiddleware()
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

    # Track the state of model_settings during handler execution
    settings_during_call = {}

    def mock_handler(req: ModelRequest) -> ModelResponse:
        # Capture model_settings during handler execution
        settings_during_call.update(req.model_settings)
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = middleware.wrap_model_call(fake_request, mock_handler)

    # Verify cache_control was present during handler execution
    assert "cache_control" in settings_during_call
    assert settings_during_call["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    # Verify cache_control is cleaned up after handler returns
    assert "cache_control" not in fake_request.model_settings
    assert fake_request.model_settings == {}
    assert isinstance(result, ModelResponse)


def test_cache_control_cleanup_on_error() -> None:
    """Test that cache_control is cleaned up even when handler raises exception.

    This ensures cleanup happens in all cases, preventing cache_control from
    persisting when fallback middleware tries alternative models.
    """
    middleware = AnthropicPromptCachingMiddleware()
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

    # Track the state of model_settings during handler execution
    settings_during_call = {}

    def failing_handler(req: ModelRequest) -> ModelResponse:
        # Capture model_settings before raising error
        settings_during_call.update(req.model_settings)
        msg = "Simulated API error"
        raise RuntimeError(msg)

    # Handler should raise the exception
    with pytest.raises(RuntimeError, match="Simulated API error"):
        middleware.wrap_model_call(fake_request, failing_handler)

    # Verify cache_control was present during handler execution
    assert "cache_control" in settings_during_call

    # Verify cache_control is cleaned up even after exception
    assert "cache_control" not in fake_request.model_settings
    assert fake_request.model_settings == {}


async def test_cache_control_cleanup_on_success_async() -> None:
    """Test async cleanup of cache_control after successful handler execution."""
    middleware = AnthropicPromptCachingMiddleware()
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

    # Track the state of model_settings during handler execution
    settings_during_call = {}

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        # Capture model_settings during handler execution
        settings_during_call.update(req.model_settings)
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)

    # Verify cache_control was present during handler execution
    assert "cache_control" in settings_during_call
    assert settings_during_call["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    # Verify cache_control is cleaned up after handler returns
    assert "cache_control" not in fake_request.model_settings
    assert fake_request.model_settings == {}
    assert isinstance(result, ModelResponse)


async def test_cache_control_cleanup_on_error_async() -> None:
    """Test async cleanup of cache_control even when handler raises exception."""
    middleware = AnthropicPromptCachingMiddleware()
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

    # Track the state of model_settings during handler execution
    settings_during_call = {}

    async def failing_handler(req: ModelRequest) -> ModelResponse:
        # Capture model_settings before raising error
        settings_during_call.update(req.model_settings)
        msg = "Simulated async API error"
        raise RuntimeError(msg)

    # Handler should raise the exception
    with pytest.raises(RuntimeError, match="Simulated async API error"):
        await middleware.awrap_model_call(fake_request, failing_handler)

    # Verify cache_control was present during handler execution
    assert "cache_control" in settings_during_call

    # Verify cache_control is cleaned up even after exception
    assert "cache_control" not in fake_request.model_settings
    assert fake_request.model_settings == {}


def test_no_cleanup_when_caching_not_applied() -> None:
    """Test that cleanup doesn't interfere when caching is not applied.

    When using an unsupported model or below min_messages_to_cache,
    cache_control should never be added or cleaned up.
    """
    middleware = AnthropicPromptCachingMiddleware(
        unsupported_model_behavior="ignore",
        min_messages_to_cache=10,
    )

    fake_request = ModelRequest(
        model=FakeToolCallingModel(),  # Unsupported model
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
        # Verify cache_control was never added
        assert "cache_control" not in req.model_settings
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = middleware.wrap_model_call(fake_request, mock_handler)

    # Verify model_settings remain empty throughout
    assert fake_request.model_settings == {}
    assert isinstance(result, ModelResponse)
