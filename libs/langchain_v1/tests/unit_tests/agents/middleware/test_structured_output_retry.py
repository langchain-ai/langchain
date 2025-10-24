"""Tests for StructuredOutputRetryMiddleware."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    StructuredOutputRetryMiddleware,
)
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    StructuredOutputError,
    StructuredOutputValidationError,
)


class WeatherReport(BaseModel):
    """Weather report schema."""

    temperature: float
    conditions: str


def test_structured_output_retry_initialization() -> None:
    """Test that StructuredOutputRetryMiddleware initializes correctly."""
    middleware = StructuredOutputRetryMiddleware()
    assert middleware.max_retries == 2

    middleware_custom = StructuredOutputRetryMiddleware(max_retries=5)
    assert middleware_custom.max_retries == 5


def test_structured_output_retry_validation_errors() -> None:
    """Test that middleware validates parameters."""
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        StructuredOutputRetryMiddleware(max_retries=-1)


def test_structured_output_retry_success_first_attempt() -> None:
    """Test that successful calls on first attempt don't trigger retry."""
    middleware = StructuredOutputRetryMiddleware(max_retries=2)

    # Create mock request and handler
    mock_request = MagicMock(spec=ModelRequest)
    mock_request.messages = []

    expected_response = ModelResponse(
        result=[AIMessage(content='{"temperature": 72.5, "conditions": "sunny"}')],
        structured_response=WeatherReport(temperature=72.5, conditions="sunny"),
    )

    handler = MagicMock(return_value=expected_response)

    # Execute
    result = middleware.wrap_model_call(mock_request, handler)

    # Verify
    assert result == expected_response
    assert handler.call_count == 1


def test_structured_output_retry_with_validation_error() -> None:
    """Test that validation errors trigger retry with error feedback."""
    middleware = StructuredOutputRetryMiddleware(max_retries=2)

    # Create mock request
    mock_request = MagicMock(spec=ModelRequest)
    mock_request.messages = []

    # First call fails, second succeeds
    ai_msg_with_error = AIMessage(content='{"temperature": "hot", "conditions": "sunny"}')
    validation_error = ValidationError.from_exception_data(
        "WeatherReport", [{"type": "float_parsing", "loc": ("temperature",), "input": "hot"}]
    )
    error = StructuredOutputValidationError("WeatherReport", validation_error, ai_msg_with_error)

    success_response = ModelResponse(
        result=[AIMessage(content='{"temperature": 72.5, "conditions": "sunny"}')],
        structured_response=WeatherReport(temperature=72.5, conditions="sunny"),
    )

    handler = MagicMock(side_effect=[error, success_response])

    # Execute
    result = middleware.wrap_model_call(mock_request, handler)

    # Verify
    assert result == success_response
    assert handler.call_count == 2

    # Check that error feedback was added to messages
    assert len(mock_request.messages) == 2
    assert mock_request.messages[0] == ai_msg_with_error
    assert isinstance(mock_request.messages[1], HumanMessage)
    assert "errors" in mock_request.messages[1].content.lower()


def test_structured_output_retry_exhausted() -> None:
    """Test that exception is raised when retries are exhausted."""
    middleware = StructuredOutputRetryMiddleware(max_retries=2)

    # Create mock request
    mock_request = MagicMock(spec=ModelRequest)
    mock_request.messages = []

    # All calls fail
    ai_msg = AIMessage(content='{"temperature": "hot", "conditions": "sunny"}')
    validation_error = ValidationError.from_exception_data(
        "WeatherReport", [{"type": "float_parsing", "loc": ("temperature",), "input": "hot"}]
    )
    error = StructuredOutputValidationError("WeatherReport", validation_error, ai_msg)

    handler = MagicMock(side_effect=error)

    # Execute and verify it raises
    with pytest.raises(StructuredOutputValidationError):
        middleware.wrap_model_call(mock_request, handler)

    # Verify handler was called 3 times (initial + 2 retries)
    assert handler.call_count == 3


def test_structured_output_retry_multiple_outputs_error() -> None:
    """Test that MultipleStructuredOutputsError triggers retry."""
    middleware = StructuredOutputRetryMiddleware(max_retries=1)

    # Create mock request
    mock_request = MagicMock(spec=ModelRequest)
    mock_request.messages = []

    # First call has multiple outputs error
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "WeatherReport", "args": {}, "id": "1"},
            {"name": "WeatherReport", "args": {}, "id": "2"},
        ],
    )
    error = MultipleStructuredOutputsError(["WeatherReport", "WeatherReport"], ai_msg)

    # Second call succeeds
    success_response = ModelResponse(
        result=[AIMessage(content='{"temperature": 72.5, "conditions": "sunny"}')],
        structured_response=WeatherReport(temperature=72.5, conditions="sunny"),
    )

    handler = MagicMock(side_effect=[error, success_response])

    # Execute
    result = middleware.wrap_model_call(mock_request, handler)

    # Verify
    assert result == success_response
    assert handler.call_count == 2






def test_structured_output_retry_ai_message_preserved() -> None:
    """Test that AI message from exception is preserved in retry messages."""
    middleware = StructuredOutputRetryMiddleware(max_retries=1)

    # Create mock request
    mock_request = MagicMock(spec=ModelRequest)
    mock_request.messages = []

    # Create specific AI message with error
    ai_msg_with_error = AIMessage(
        content='{"temperature": "invalid", "conditions": "sunny"}',
        id="test-id-123",
    )
    error = StructuredOutputValidationError("WeatherReport", ValueError("Invalid"), ai_msg_with_error)

    success_response = ModelResponse(
        result=[AIMessage(content='{"temperature": 72.5, "conditions": "sunny"}')],
        structured_response=WeatherReport(temperature=72.5, conditions="sunny"),
    )

    handler = MagicMock(side_effect=[error, success_response])

    # Execute
    result = middleware.wrap_model_call(mock_request, handler)

    # Verify the original AI message with error was added to messages
    assert len(mock_request.messages) == 2
    assert mock_request.messages[0] == ai_msg_with_error
    assert mock_request.messages[0].id == "test-id-123"


async def test_structured_output_retry_async() -> None:
    """Test async version of retry middleware."""
    middleware = StructuredOutputRetryMiddleware(max_retries=1)

    # Create mock request
    mock_request = MagicMock(spec=ModelRequest)
    mock_request.messages = []

    # First call fails
    ai_msg = AIMessage(content='{"temperature": "hot"}')
    error = StructuredOutputValidationError("WeatherReport", ValueError("Invalid"), ai_msg)

    # Second call succeeds
    success_response = ModelResponse(
        result=[AIMessage(content='{"temperature": 72.5, "conditions": "sunny"}')],
        structured_response=WeatherReport(temperature=72.5, conditions="sunny"),
    )

    async def async_handler(request: ModelRequest) -> ModelResponse:
        if not hasattr(async_handler, "call_count"):
            async_handler.call_count = 0  # type: ignore[attr-defined]
        async_handler.call_count += 1  # type: ignore[attr-defined]

        if async_handler.call_count == 1:  # type: ignore[attr-defined]
            raise error
        return success_response

    # Execute
    result = await middleware.awrap_model_call(mock_request, async_handler)

    # Verify
    assert result == success_response
    assert async_handler.call_count == 2  # type: ignore[attr-defined]
