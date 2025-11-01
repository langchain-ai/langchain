"""Tests for OpenAI prompt caching middleware."""

import warnings
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.middleware import OpenAIPromptCachingMiddleware


def test_openai_prompt_caching_middleware_initialization() -> None:
    """Test OpenAIPromptCachingMiddleware initialization."""
    middleware = OpenAIPromptCachingMiddleware(
        type="ephemeral",
        ttl="1h",
        min_messages_to_cache=5,
        unsupported_model_behavior="ignore",
    )
    assert middleware.type == "ephemeral"
    assert middleware.ttl == "1h"
    assert middleware.min_messages_to_cache == 5
    assert middleware.unsupported_model_behavior == "ignore"

    # Test with default values
    middleware = OpenAIPromptCachingMiddleware()
    assert middleware.type == "ephemeral"
    assert middleware.ttl == "5m"
    assert middleware.min_messages_to_cache == 0
    assert middleware.unsupported_model_behavior == "warn"


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
        runtime=Runtime(),
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


def test_openai_prompt_caching_middleware_unsupported_model() -> None:
    """Test OpenAIPromptCachingMiddleware with unsupported model."""
    fake_model = MagicMock()  # Not a ChatOpenAI instance

    fake_request = ModelRequest(
        model=fake_model,
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=Runtime(),
        model_settings={},
    )

    middleware = OpenAIPromptCachingMiddleware(unsupported_model_behavior="raise")

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Test that it raises an error for unsupported model instances
    with pytest.raises(ValueError):
        middleware.wrap_model_call(fake_request, mock_handler)

    # Test warn behavior
    middleware = OpenAIPromptCachingMiddleware(unsupported_model_behavior="warn")
    with warnings.catch_warnings(record=True):
        result = middleware.wrap_model_call(fake_request, mock_handler)
        assert isinstance(result, ModelResponse)

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
        runtime=Runtime(),
        model_settings={},
    )

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = await middleware.awrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Check that model_settings were updated
    assert fake_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "1h"}
    }


def test_openai_prompt_caching_middleware_min_messages() -> None:
    """Test that middleware respects min_messages_to_cache."""
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
        runtime=Runtime(),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should NOT be added when message count is below minimum
    assert fake_request.model_settings == {}


def test_openai_prompt_caching_middleware_with_system_prompt() -> None:
    """Test that system prompt is counted in message count."""
    middleware = OpenAIPromptCachingMiddleware(min_messages_to_cache=3)

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
        runtime=Runtime(),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="mock response")])

    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should be added when system prompt pushes count to minimum
    assert fake_request.model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "5m"}
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
        runtime=Runtime(),
        model_settings={},
    )

    result = middleware.wrap_model_call(fake_request_no_system, mock_handler)
    assert isinstance(result, ModelResponse)
    # Cache control should NOT be added since there are only 2 messages
    assert fake_request_no_system.model_settings == {}
