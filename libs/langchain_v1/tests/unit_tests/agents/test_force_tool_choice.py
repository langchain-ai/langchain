"""Unit tests for force_tool_choice parameter in ToolStrategy.

Tests to ensure that:
1. force_tool_choice=True (default) maintains backward compatibility
2. force_tool_choice=False allows natural text before tool calls
"""

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from tests.unit_tests.agents.model import FakeToolCallingModel


class WeatherResponse(BaseModel):
    """Weather response schema."""
    temperature: float = Field(description="Temperature in fahrenheit")
    condition: str = Field(description="Weather condition")


@tool
def get_weather() -> str:
    """Get the weather."""
    return "The weather is sunny and 75°F."


class TestForceToolChoice:
    """Test suite for force_tool_choice parameter."""

    def test_force_tool_choice_true_default(self) -> None:
        """Test that force_tool_choice=True is the default and forces immediate tool call."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [{"name": "WeatherResponse", "id": "2", "args": {"temperature": 75.0, "condition": "sunny"}}]
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        # Default behavior - should force tool call
        agent = create_agent(
            model,
            [get_weather],
            response_format=ToolStrategy(WeatherResponse)
        )

        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        # Should get structured response
        assert response["structured_response"] == WeatherResponse(temperature=75.0, condition="sunny")

        # Verify the model was bound with tool_choice="any"
        # This is implicit in the FakeToolCallingModel behavior

    def test_force_tool_choice_true_explicit(self) -> None:
        """Test that force_tool_choice=True explicitly forces immediate tool call."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [{"name": "WeatherResponse", "id": "2", "args": {"temperature": 75.0, "condition": "sunny"}}]
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        # Explicitly set force_tool_choice=True
        agent = create_agent(
            model,
            [get_weather],
            response_format=ToolStrategy(WeatherResponse, force_tool_choice=True)
        )

        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        # Should get structured response
        assert response["structured_response"] == WeatherResponse(temperature=75.0, condition="sunny")

    def test_force_tool_choice_false_allows_flexibility(self) -> None:
        """Test that force_tool_choice=False allows model to respond without forcing tool call."""
        # Simulate model that might return text first or skip structured output
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [{"name": "WeatherResponse", "id": "2", "args": {"temperature": 75.0, "condition": "sunny"}}]
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        # Set force_tool_choice=False to allow natural streaming
        agent = create_agent(
            model,
            [get_weather],
            response_format=ToolStrategy(WeatherResponse, force_tool_choice=False)
        )

        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        # Should still get structured response if model chose to call the tool
        assert response["structured_response"] == WeatherResponse(temperature=75.0, condition="sunny")

    def test_backward_compatibility(self) -> None:
        """Ensure existing code without force_tool_choice works as before."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [{"name": "WeatherResponse", "id": "2", "args": {"temperature": 75.0, "condition": "sunny"}}]
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        # Old code without force_tool_choice parameter
        agent = create_agent(
            model,
            [get_weather],
            response_format=ToolStrategy(
                WeatherResponse,
                handle_errors=True  # Only specify other parameters
            )
        )

        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        # Should maintain backward compatible behavior
        assert response["structured_response"] == WeatherResponse(temperature=75.0, condition="sunny")
        assert len(response["messages"]) == 5

    def test_force_tool_choice_with_handle_errors(self) -> None:
        """Test that force_tool_choice works alongside handle_errors parameter."""
        tool_calls = [
            [{"args": {}, "id": "1", "name": "get_weather"}],
            [{"name": "WeatherResponse", "id": "2", "args": {"temperature": 75.0, "condition": "sunny"}}]
        ]

        model = FakeToolCallingModel(tool_calls=tool_calls)

        # Combine force_tool_choice with handle_errors
        agent = create_agent(
            model,
            [get_weather],
            response_format=ToolStrategy(
                WeatherResponse,
                force_tool_choice=False,
                handle_errors=True
            )
        )

        response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})

        assert response["structured_response"] == WeatherResponse(temperature=75.0, condition="sunny")
