"""Tests for NoStructuredOutputError functionality (issue #36349)."""

from collections.abc import Callable

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.structured_output import (
    NoStructuredOutputError,
    StructuredOutputError,
    ToolStrategy,
)
from tests.unit_tests.agents.model import FakeToolCallingModel


class WeatherReport(BaseModel):
    """Weather report schema for testing."""

    temperature: float
    conditions: str


@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city.

    Args:
        city: The city to get weather for.

    Returns:
        Weather information for the city.
    """
    return f"The weather in {city} is sunny and 72 degrees."


def test_no_structured_output_error_raised() -> None:
    """Test that NoStructuredOutputError is raised when model makes no tool calls."""
    # Model returns no tool calls at all
    tool_calls = [
        [],  # No tool calls
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)

    agent = create_agent(
        model=model,
        tools=[get_weather],
        response_format=ToolStrategy(schema=WeatherReport, handle_errors=False),
    )

    # Should raise NoStructuredOutputError when no tool calls are made
    with pytest.raises(NoStructuredOutputError):
        agent.invoke(
            {"messages": [HumanMessage("What's the weather in Tokyo?")]},
        )

    # Verify the model was called once
    assert model.index == 1


def test_no_structured_output_error_message() -> None:
    """Test that NoStructuredOutputError contains the AI message."""
    tool_calls = [
        [],  # No tool calls
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)

    agent = create_agent(
        model=model,
        tools=[get_weather],
        response_format=ToolStrategy(schema=WeatherReport, handle_errors=False),
    )

    try:
        agent.invoke(
            {"messages": [HumanMessage("What's the weather in Tokyo?")]},
        )
        pytest.fail("Expected NoStructuredOutputError to be raised")
    except NoStructuredOutputError as e:
        # Verify the error message
        assert "No structured output tool called" in str(e)
        # Verify the AI message is accessible
        assert e.ai_message is not None


def test_no_structured_output_with_retry() -> None:
    """Test that no structured output error triggers retry when handle_errors=True."""
    # First attempt: no tool calls, second attempt: valid structured output
    tool_calls = [
        [],  # No tool calls on first attempt
        [
            {
                "name": "WeatherReport",
                "id": "1",
                "args": {"temperature": 72.5, "conditions": "sunny"},
            }
        ],  # Valid on second attempt
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)

    agent = create_agent(
        model=model,
        tools=[get_weather],
        response_format=ToolStrategy(schema=WeatherReport, handle_errors=True),
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather in Tokyo?")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify we got a structured response after retry
    assert "structured_response" in result
    structured = result["structured_response"]
    assert isinstance(structured, WeatherReport)
    assert structured.temperature == 72.5
    assert structured.conditions == "sunny"

    # Verify the model was called twice (initial + 1 retry)
    assert model.index == 2


def test_no_structured_output_is_structured_output_error() -> None:
    """Test that NoStructuredOutputError is a subclass of StructuredOutputError."""
    assert issubclass(NoStructuredOutputError, StructuredOutputError)


class StructuredOutputRetryMiddleware(AgentMiddleware):
    """Retries model calls when structured output parsing fails."""

    def __init__(self, max_retries: int) -> None:
        """Initialize the structured output retry middleware.

        Args:
            max_retries: Maximum number of retry attempts.
        """
        self.max_retries = max_retries

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Intercept and control model execution via handler callback.

        Args:
            request: The model request containing messages and configuration.
            handler: The function to call the model.

        Returns:
            The model response.

        Raises:
            StructuredOutputError: If max retries exceeded without success.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except StructuredOutputError as exc:
                if attempt == self.max_retries:
                    raise

                # Include both the AI message and error in a single human message
                # to maintain valid chat history alternation
                ai_content = exc.ai_message.content
                error_message = (
                    f"Your previous response was:\\n{ai_content}\\n\\n"
                    f"Error: {exc}. Please try again with a valid response."
                )
                request.messages.append(HumanMessage(content=error_message))

        # This should never be reached, but satisfies type checker
        return handler(request)


def test_no_structured_output_retry_with_middleware() -> None:
    """Test retry middleware catches NoStructuredOutputError."""
    # First two attempts: no tool calls, third attempt: valid
    tool_calls = [
        [],  # No tool calls on first attempt
        [],  # No tool calls on second attempt
        [
            {
                "name": "WeatherReport",
                "id": "1",
                "args": {"temperature": 75.0, "conditions": "cloudy"},
            }
        ],  # Valid on third attempt
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)
    retry_middleware = StructuredOutputRetryMiddleware(max_retries=2)

    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[retry_middleware],
        response_format=ToolStrategy(schema=WeatherReport, handle_errors=False),
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather in London?")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify we got a structured response after retries
    assert "structured_response" in result
    structured = result["structured_response"]
    assert isinstance(structured, WeatherReport)
    assert structured.temperature == 75.0
    assert structured.conditions == "cloudy"

    # Verify the model was called 3 times (initial + 2 retries)
    assert model.index == 3


def test_no_structured_output_retry_exceeds_max() -> None:
    """Test that retry middleware raises error when max retries exceeded."""
    # All attempts: no tool calls
    tool_calls = [
        [],
        [],
        [],
    ]

    model = FakeToolCallingModel(tool_calls=tool_calls)
    retry_middleware = StructuredOutputRetryMiddleware(max_retries=2)

    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[retry_middleware],
        response_format=ToolStrategy(schema=WeatherReport, handle_errors=False),
        checkpointer=InMemorySaver(),
    )

    # Should raise NoStructuredOutputError after exhausting retries
    with pytest.raises(NoStructuredOutputError):
        agent.invoke(
            {"messages": [HumanMessage("What's the weather in Paris?")]},
            {"configurable": {"thread_id": "test"}},
        )

    # Verify the model was called 3 times (initial + 2 retries)
    assert model.index == 3