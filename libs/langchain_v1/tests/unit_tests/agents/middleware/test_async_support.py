"""Test async support for middleware classes."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
from langchain.agents.middleware.planning import PlanningMiddleware
from langchain.agents.middleware.prompt_caching import (
    AnthropicPromptCachingMiddleware,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse


class TestAsyncMiddlewareSupport:
    """Test async support for all middleware classes."""

    @pytest.mark.asyncio
    async def test_planning_middleware_async(self) -> None:
        """Test that PlanningMiddleware.awrap_model_call works correctly."""
        middleware = PlanningMiddleware()

        # Create mock request and handler
        mock_request = MagicMock(spec=ModelRequest)
        mock_request.system_prompt = "Original prompt"
        mock_request.messages = []
        mock_request.state = {"messages": []}

        mock_response = ModelResponse(result=[AIMessage(content="Test response")])
        async_handler = AsyncMock(return_value=mock_response)

        # Test async execution
        result = await middleware.awrap_model_call(mock_request, async_handler)

        # Verify the system prompt was updated
        assert middleware.system_prompt in mock_request.system_prompt
        # Verify handler was called
        async_handler.assert_awaited_once_with(mock_request)
        # Verify result
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_anthropic_caching_middleware_async(self) -> None:
        """Test that AnthropicPromptCachingMiddleware.awrap_model_call works correctly."""
        middleware = AnthropicPromptCachingMiddleware(
            min_messages_to_cache=2,
            unsupported_model_behavior="ignore",
        )

        # Create mock request and handler
        mock_request = MagicMock(spec=ModelRequest)
        mock_request.model = MagicMock()  # Non-Anthropic model for simplicity
        mock_request.system_prompt = "Test prompt"
        mock_request.messages = ["msg1", "msg2", "msg3"]  # Enough messages to cache
        mock_request.model_settings = {}

        mock_response = ModelResponse(result=[AIMessage(content="Test response")])
        async_handler = AsyncMock(return_value=mock_response)

        # Test async execution
        result = await middleware.awrap_model_call(mock_request, async_handler)

        # Verify handler was called
        async_handler.assert_awaited_once_with(mock_request)
        # Verify result
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_model_fallback_middleware_async_success(self) -> None:
        """Test ModelFallbackMiddleware.awrap_model_call with successful primary model."""
        # Create mock fallback models
        fallback1 = MagicMock()
        fallback2 = MagicMock()

        middleware = ModelFallbackMiddleware(fallback1, fallback2)

        # Create mock request and handler
        mock_request = MagicMock(spec=ModelRequest)
        mock_request.model = MagicMock()  # Primary model

        mock_response = ModelResponse(result=[AIMessage(content="Primary response")])
        async_handler = AsyncMock(return_value=mock_response)

        # Test async execution - primary succeeds
        result = await middleware.awrap_model_call(mock_request, async_handler)

        # Verify only called once (primary succeeded)
        async_handler.assert_awaited_once()
        # Verify result
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_model_fallback_middleware_async_fallback(self) -> None:
        """Test ModelFallbackMiddleware.awrap_model_call with fallback logic."""
        # Create mock fallback models
        fallback1 = MagicMock()
        fallback2 = MagicMock()

        middleware = ModelFallbackMiddleware(fallback1, fallback2)

        # Create mock request and handler
        mock_request = MagicMock(spec=ModelRequest)
        mock_request.model = MagicMock()  # Primary model

        # Create handler that fails for primary and fallback1, succeeds for fallback2
        call_count = 0

        async def handler(request: Any) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail for primary and fallback1
                raise Exception(f"Model {call_count} failed")
            return ModelResponse(result=[AIMessage(content=f"Response {call_count}")])

        # Test async execution with fallback
        result = await middleware.awrap_model_call(mock_request, handler)

        # Verify called 3 times (primary + fallback1 + fallback2)
        assert call_count == 3
        # Verify result is from third attempt (fallback2)
        assert isinstance(result, ModelResponse)
        assert result.result[0].content == "Response 3"
        # Verify model was updated to fallback2
        assert mock_request.model == fallback2

    @pytest.mark.asyncio
    async def test_model_fallback_middleware_async_all_fail(self) -> None:
        """Test ModelFallbackMiddleware.awrap_model_call when all models fail."""
        # Create mock fallback model
        fallback1 = MagicMock()

        middleware = ModelFallbackMiddleware(fallback1)

        # Create mock request and handler
        mock_request = MagicMock(spec=ModelRequest)
        mock_request.model = MagicMock()  # Primary model

        # Create handler that always fails
        async def handler(request: Any) -> ModelResponse:
            raise ValueError("All models fail")

        # Test async execution - all fail
        with pytest.raises(ValueError, match="All models fail"):
            await middleware.awrap_model_call(mock_request, handler)


def test_all_middleware_have_awrap_model_call() -> None:
    """Verify all middleware classes that use wrap_model_call also have awrap_model_call."""
    # Test PlanningMiddleware
    planning_middleware = PlanningMiddleware()
    assert hasattr(planning_middleware, "wrap_model_call")
    assert hasattr(planning_middleware, "awrap_model_call")
    # Check it's not the base class implementation
    planning_method = getattr(planning_middleware.__class__, "awrap_model_call")
    assert "planning" in planning_method.__module__

    # Test AnthropicPromptCachingMiddleware
    caching_middleware = AnthropicPromptCachingMiddleware()
    assert hasattr(caching_middleware, "wrap_model_call")
    assert hasattr(caching_middleware, "awrap_model_call")
    # Check it's not the base class implementation
    caching_method = getattr(caching_middleware.__class__, "awrap_model_call")
    assert "prompt_caching" in caching_method.__module__

    # Test ModelFallbackMiddleware
    fallback_middleware = ModelFallbackMiddleware(MagicMock())
    assert hasattr(fallback_middleware, "wrap_model_call")
    assert hasattr(fallback_middleware, "awrap_model_call")
    # Check it's not the base class implementation
    fallback_method = getattr(fallback_middleware.__class__, "awrap_model_call")
    assert "model_fallback" in fallback_method.__module__