"""Tests for AnthropicPromptCachingMiddleware."""

import warnings
from types import ModuleType
from unittest.mock import patch

import pytest

from langchain_core.messages import HumanMessage
from langchain.agents.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langchain.agents.middleware.types import ModelRequest

from ..model import FakeToolCallingModel


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
        model_settings={},
    )

    assert middleware.modify_model_request(fake_request, {}, None).model_settings == {
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
        model_settings={},
    )

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="raise")

    with pytest.raises(
        ValueError,
        match="AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models. Please install langchain-anthropic.",
    ):
        middleware.modify_model_request(fake_request, {}, None)

    langchain_anthropic = ModuleType("langchain_anthropic")

    class MockChatAnthropic:
        pass

    langchain_anthropic.ChatAnthropic = MockChatAnthropic

    with patch.dict("sys.modules", {"langchain_anthropic": langchain_anthropic}):
        with pytest.raises(
            ValueError,
            match="AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models, not instances of",
        ):
            middleware.modify_model_request(fake_request, {}, None)

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="warn")

    with warnings.catch_warnings(record=True) as w:
        result = middleware.modify_model_request(fake_request, {}, None)
        assert len(w) == 1
        assert (
            "AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models. Please install langchain-anthropic."
            in str(w[-1].message)
        )
        assert result == fake_request

    with warnings.catch_warnings(record=True) as w:
        with patch.dict("sys.modules", {"langchain_anthropic": langchain_anthropic}):
            result = middleware.modify_model_request(fake_request, {}, None)
            assert result is fake_request
            assert len(w) == 1
            assert (
                "AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models, not instances of"
                in str(w[-1].message)
            )

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")

    result = middleware.modify_model_request(fake_request, {}, None)
    assert result is fake_request

    with patch.dict("sys.modules", {"langchain_anthropic": {"ChatAnthropic": object()}}):
        result = middleware.modify_model_request(fake_request, {}, None)
        assert result is fake_request
