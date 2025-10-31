"""Tests for OpenAI prompt caching middleware."""

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

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.middleware import OpenAIPromptCachingMiddleware


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


def test_openai_prompt_caching_middleware_initialization() -> None:
    """Test OpenAIPromptCachingMiddleware initialization."""
    # Test with custom values
    middleware = OpenAIPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=5
    )
    assert middleware.type == "ephemeral"
    assert middleware.ttl == "1h"
    assert middleware.min_messages_to_cache == 5

    # Test with default values
    middleware = OpenAIPromptCachingMiddleware()
    assert middleware.type == "ephemeral"
    assert middleware.ttl == "5m"
    assert middleware.min_messages_to_cache == 0

    # Test with different unsupported_model_behavior values
    middleware = OpenAIPromptCachingMiddleware(unsupported_model_behavior="ignore")
    assert middleware.unsupported_model_behavior == "ignore"

    middleware = OpenAIPromptCachingMiddleware(unsupported_model_behavior="warn")
    assert middleware.unsupported_model_behavior == "warn"

    middleware = OpenAIPromptCachingMiddleware(unsupported_model_behavior="raise")
    assert middleware.unsupported_model_behavior == "raise"


def test_openai_prompt_caching_middleware_sync_with_openai_model() -> None:
    """Test OpenAIPromptCachingMiddleware sync path with OpenAI model."""
    middleware = OpenAIPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=0
    )

    # Create a mock ChatOpenAI instance
    mock_chat_openai = MagicMock(spec=ChatOpenAI)

    fake_request = ModelRequest(
        model=mock_chat_openai,
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

    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    
    # Check that cache control was added to model_settings
    assert fake_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "1h"}
    }


def test_openai_prompt_caching_middleware_sync_unsupported_model() -> None:
    """Test OpenAIPromptCachingMiddleware with unsupported model."""
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

    middleware = OpenAIPromptCachingMiddleware(unsupported_model_behavior="raise")

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Test that it raises an error for unsupported model instances
    with pytest.raises(
        ValueError,
        match=(
            "OpenAIPromptCachingMiddleware caching middleware only supports "
            "OpenAI models, not instances of"
        ),
    ):
        middleware.wrap_model_call(fake_request, mock_handler)

    middleware = OpenAIPromptCachingMiddleware(unsupported_model_behavior="warn")

    # Test warn behavior for unsupported model instances
    with warnings.catch_warnings(record=True) as w:
        result = middleware.wrap_model_call(fake_request, mock_handler)
        assert isinstance(result, ModelResponse)
        assert len(w) == 1
        assert (
            "OpenAIPromptCachingMiddleware caching middleware only supports "
            "OpenAI models, not instances of"
        ) in str(w[-1].message)

    # Test ignore behavior
    middleware = OpenAIPromptCachingMiddleware(unsupported_model_behavior="ignore")
    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should NOT be added when ignoring unsupported models
    assert fake_request.model_settings == {}


async def test_openai_prompt_caching_middleware_async_with_openai_model() -> None:
    """Test OpenAIPromptCachingMiddleware async path with OpenAI model."""
    middleware = OpenAIPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=0
    )

    # Create a mock ChatOpenAI instance
    mock_chat_openai = MagicMock(spec=ChatOpenAI)

    fake_request = ModelRequest(
        model=mock_chat_openai,
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
    # Check that model_settings were passed through via the request
    assert fake_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "1h"}
    }


async def test_openai_prompt_caching_middleware_async_unsupported_model() -> None:
    """Test OpenAIPromptCachingMiddleware async path with unsupported model."""
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

    middleware = OpenAIPromptCachingMiddleware(unsupported_model_behavior="raise")

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Test that it raises an error for unsupported model instances
    with pytest.raises(
        ValueError,
        match=(
            "OpenAIPromptCachingMiddleware caching middleware only supports "
            "OpenAI models, not instances of"
        ),
    ):
        await middleware.awrap_model_call(fake_request, mock_handler)

    middleware = OpenAIPromptCachingMiddleware(unsupported_model_behavior="warn")

    # Test warn behavior for unsupported model instances
    with warnings.catch_warnings(record=True) as w:
        result = await middleware.awrap_model_call(fake_request, mock_handler)
        assert isinstance(result, ModelResponse)
        assert len(w) == 1
        assert (
            "OpenAIPromptCachingMiddleware caching middleware only supports "
            "OpenAI models, not instances of"
        ) in str(w[-1].message)

    # Test ignore behavior
    middleware = OpenAIPromptCachingMiddleware(unsupported_model_behavior="ignore")
    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should NOT be added when ignoring unsupported models
    assert fake_request.model_settings == {}


def test_openai_prompt_caching_middleware_sync_min_messages() -> None:
    """Test sync path respects min_messages_to_cache."""
    middleware = OpenAIPromptCachingMiddleware(min_messages_to_cache=5)

    # Create a mock ChatOpenAI instance
    mock_chat_openai = MagicMock(spec=ChatOpenAI)

    # Test with fewer messages than minimum
    fake_request = ModelRequest(
        model=mock_chat_openai,
        messages=[HumanMessage("Hello")] * 3,
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")] * 3},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should NOT be added when message count is below minimum
    assert fake_request.model_settings == {}


async def test_openai_prompt_caching_middleware_async_min_messages() -> None:
    """Test async path respects min_messages_to_cache."""
    middleware = OpenAIPromptCachingMiddleware(min_messages_to_cache=5)

    # Create a mock ChatOpenAI instance
    mock_chat_openai = MagicMock(spec=ChatOpenAI)

    # Test with fewer messages than minimum
    fake_request = ModelRequest(
        model=mock_chat_openai,
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


def test_openai_prompt_caching_middleware_sync_with_system_prompt() -> None:
    """Test sync path counts system prompt in message count."""
    middleware = OpenAIPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=3
    )

    # Create a mock ChatOpenAI instance
    mock_chat_openai = MagicMock(spec=ChatOpenAI)

    # Test with system prompt: 2 messages + 1 system = 3 total (meets minimum)
    fake_request = ModelRequest(
        model=mock_chat_openai,
        messages=[HumanMessage("Hello"), HumanMessage("World")],
        system_prompt="You are a helpful assistant",
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello"), HumanMessage("World")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should be added when system prompt pushes count to minimum
    assert fake_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "1h"}
    }

    # Test without system prompt - should not trigger caching
    fake_request_no_system = ModelRequest(
        model=mock_chat_openai,
        messages=[HumanMessage("Hello"), HumanMessage("World")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello"), HumanMessage("World")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    result = middleware.wrap_model_call(fake_request_no_system, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should NOT be added since there are only 2 messages and system_prompt is None
    assert fake_request_no_system.model_settings == {}


async def test_openai_prompt_caching_middleware_async_with_system_prompt() -> None:
    """Test async path counts system prompt in message count."""
    middleware = OpenAIPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=3
    )

    # Create a mock ChatOpenAI instance
    mock_chat_openai = MagicMock(spec=ChatOpenAI)

    # Test with system prompt: 2 messages + 1 system = 3 total (meets minimum)
    fake_request = ModelRequest(
        model=mock_chat_openai,
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

    # Test without system prompt - should not trigger caching
    fake_request_no_system = ModelRequest(
        model=mock_chat_openai,
        messages=[HumanMessage("Hello"), HumanMessage("World")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello"), HumanMessage("World")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    result = await middleware.awrap_model_call(fake_request_no_system, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should NOT be added since there are only 2 messages and system_prompt is None
    assert fake_request_no_system.model_settings == {}


def test_openai_prompt_caching_middleware_sync_default_values() -> None:
    """Test sync path with default middleware initialization."""
    # Test with default values (min_messages_to_cache=0)
    middleware = OpenAIPromptCachingMiddleware()

    # Create a mock ChatOpenAI instance
    mock_chat_openai = MagicMock(spec=ChatOpenAI)

    # Single message should trigger caching with default settings
    fake_request = ModelRequest(
        model=mock_chat_openai,
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

    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Check that model_settings were added with default values
    assert fake_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "5m"}
    }


async def test_openai_prompt_caching_middleware_async_default_values() -> None:
    """Test async path with default middleware initialization."""
    # Test with default values (min_messages_to_cache=0)
    middleware = OpenAIPromptCachingMiddleware()

    # Create a mock ChatOpenAI instance
    mock_chat_openai = MagicMock(spec=ChatOpenAI)

    # Single message should trigger caching with default settings
    fake_request = ModelRequest(
        model=mock_chat_openai,
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


def test_openai_prompt_caching_middleware_should_apply_caching() -> None:
    """Test the _should_apply_caching method directly."""
    middleware = OpenAIPromptCachingMiddleware(min_messages_to_cache=2)

    # Create a mock ChatOpenAI instance
    mock_chat_openai = MagicMock(spec=ChatOpenAI)

    # Test with enough messages
    fake_request = ModelRequest(
        model=mock_chat_openai,
        messages=[HumanMessage("Hello"), HumanMessage("World")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello"), HumanMessage("World")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    # Should return True when model is ChatOpenAI and message count >= min
    assert middleware._should_apply_caching(fake_request) is True

    # Test with not enough messages
    fake_request_few = ModelRequest(
        model=mock_chat_openai,
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    assert middleware._should_apply_caching(fake_request_few) is False

    # Test with unsupported model
    fake_request_unsupported = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello"), HumanMessage("World")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello"), HumanMessage("World")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    assert middleware._should_apply_caching(fake_request_unsupported) is False

    # Test with system prompt included in count
    middleware_with_system = OpenAIPromptCachingMiddleware(min_messages_to_cache=3)
    fake_request_system = ModelRequest(
        model=mock_chat_openai,
        messages=[HumanMessage("Hello")],  # 1 message
        system_prompt="You are a helpful assistant",  # +1 for system = 2 total
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    # Should return False since 1 message + 1 system = 2 < 3
    assert middleware_with_system._should_apply_caching(fake_request_system) is False

    # Test with system prompt included in count that meets threshold
    middleware_with_system2 = OpenAIPromptCachingMiddleware(min_messages_to_cache=2)
    # 1 message + 1 system = 2 total which meets the minimum
    assert middleware_with_system2._should_apply_caching(fake_request_system) is True