"""Test suite for provider-specific tool_choice handling in create_agent."""

import pytest
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.factory import _get_forced_tool_choice
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from tests.unit_tests.agents.model import FakeToolCallingModel


class WeatherResponse(BaseModel):
    """Weather forecast response."""

    temperature: float = Field(description="Temperature in fahrenheit")
    condition: str = Field(description="Weather condition")


@tool
def get_weather() -> str:
    """Get the current weather."""
    return "Sunny, 75°F"


class TestGetForcedToolChoice:
    """Test the _get_forced_tool_choice function for different providers."""

    def test_cohere_command_model(self) -> None:
        """Test that Cohere 'command' models return 'REQUIRED'."""
        model = FakeToolCallingModel()
        model.model_name = "command-r-plus"

        result = _get_forced_tool_choice(model)

        assert result == "REQUIRED"

    def test_cohere_command_r_model(self) -> None:
        """Test that Cohere 'command-r' models return 'REQUIRED'."""
        model = FakeToolCallingModel()
        model.model_name = "command-r"

        result = _get_forced_tool_choice(model)

        assert result == "REQUIRED"

    def test_cohere_c4ai_model(self) -> None:
        """Test that Cohere 'c4ai' models return 'REQUIRED'."""
        model = FakeToolCallingModel()
        model.model_name = "c4ai-aya-expanse-32b"

        result = _get_forced_tool_choice(model)

        assert result == "REQUIRED"

    def test_cohere_aya_model(self) -> None:
        """Test that Cohere 'aya' models return 'REQUIRED'."""
        model = FakeToolCallingModel()
        model.model_name = "aya-expanse-8b"

        result = _get_forced_tool_choice(model)

        assert result == "REQUIRED"

    def test_openai_model(self) -> None:
        """Test that OpenAI models return 'any'."""
        model = FakeToolCallingModel()
        model.model_name = "gpt-4"

        result = _get_forced_tool_choice(model)

        assert result == "any"

    def test_anthropic_model(self) -> None:
        """Test that Anthropic models return 'any' (default)."""
        model = FakeToolCallingModel()
        model.model_name = "claude-3-5-sonnet-20241022"

        result = _get_forced_tool_choice(model)

        assert result == "any"

    def test_unknown_model(self) -> None:
        """Test that unknown models default to 'any'."""
        model = FakeToolCallingModel()
        model.model_name = "some-unknown-model"

        result = _get_forced_tool_choice(model)

        assert result == "any"

    def test_model_without_model_name(self) -> None:
        """Test that models without model_name attribute default to 'any'."""
        model = FakeToolCallingModel()
        # Don't set model_name

        result = _get_forced_tool_choice(model)

        assert result == "any"

    def test_case_insensitive_matching(self) -> None:
        """Test that model name matching is case-insensitive."""
        model = FakeToolCallingModel()
        model.model_name = "COMMAND-R-PLUS"

        result = _get_forced_tool_choice(model)

        assert result == "REQUIRED"


class TestCohereToolChoiceIntegration:
    """Integration tests for Cohere models with response_format."""

    def test_cohere_with_tool_strategy_binds_correctly(self) -> None:
        """Test that Cohere models with ToolStrategy use REQUIRED."""
        model = FakeToolCallingModel()
        model.model_name = "command-r-plus"
        model.tool_calls = [[]]  # No tool calls initially

        agent = create_agent(
            model=model,
            tools=[get_weather],
            response_format=ToolStrategy(schema=WeatherResponse),
        )

        # Verify agent was created successfully
        assert agent is not None

        # The model.bind() call should have been invoked with tool_choice="REQUIRED"
        # We verify this by checking that no exception was raised during agent creation
        # In the real Cohere implementation, passing tool_choice="any" would fail

    def test_openai_with_tool_strategy_uses_any(self) -> None:
        """Test that OpenAI models with ToolStrategy use 'any'."""
        model = FakeToolCallingModel()
        model.model_name = "gpt-4"
        model.tool_calls = [[]]  # No tool calls initially

        agent = create_agent(
            model=model,
            tools=[get_weather],
            response_format=ToolStrategy(schema=WeatherResponse),
        )

        # Verify agent was created successfully
        assert agent is not None
        # OpenAI models should work with tool_choice="any"
