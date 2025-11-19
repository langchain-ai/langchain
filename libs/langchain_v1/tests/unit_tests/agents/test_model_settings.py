"""Tests for model_settings parameter in create_agent."""

import pytest

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from .model import FakeToolCallingModel


def test_create_agent_with_model_settings_sync() -> None:
    """Test that model_settings are correctly passed when creating an agent (sync).

    Verifies that the model_settings dict provided to create_agent is properly
    passed through to the model's bind_tools method during agent execution.
    """
    # Track what kwargs were passed to bind_tools
    bind_kwargs_captured = {}

    class ModelSettingsTrackingModel(FakeToolCallingModel):
        """Model that captures kwargs passed to bind_tools."""

        def bind_tools(self, tools, **kwargs):
            bind_kwargs_captured.update(kwargs)
            return super().bind_tools(tools, **kwargs)

    # Create a simple tool
    def sample_tool(x: int) -> int:
        """A simple test tool."""
        return x * 2

    # Define model settings
    model_settings = {"temperature": 0.7, "max_tokens": 1000}

    # Create model with no tool calls (to avoid tool execution)
    model = ModelSettingsTrackingModel(tool_calls=[[]])

    # Create agent with model_settings
    agent = create_agent(
        model=model,
        tools=[sample_tool],
        model_settings=model_settings,
    )

    # Invoke the agent
    result = agent.invoke({"messages": [HumanMessage("Test message")]})

    # Verify model_settings were passed to bind_tools
    assert "temperature" in bind_kwargs_captured
    assert bind_kwargs_captured["temperature"] == 0.7
    assert "max_tokens" in bind_kwargs_captured
    assert bind_kwargs_captured["max_tokens"] == 1000

    # Verify the agent executed successfully
    assert "messages" in result
    assert len(result["messages"]) > 0


async def test_create_agent_with_model_settings_async() -> None:
    """Test that model_settings are correctly passed when creating an agent (async).

    Verifies that the model_settings dict provided to create_agent is properly
    passed through to the model's bind_tools method during async agent execution.
    """
    # Track what kwargs were passed to bind_tools
    bind_kwargs_captured = {}

    class ModelSettingsTrackingModel(FakeToolCallingModel):
        """Model that captures kwargs passed to bind_tools."""

        def bind_tools(self, tools, **kwargs):
            bind_kwargs_captured.update(kwargs)
            return super().bind_tools(tools, **kwargs)

    # Create a simple tool
    def sample_tool(query: str) -> str:
        """A simple test tool."""
        return f"Result for: {query}"

    # Define model settings with various parameters
    model_settings = {
        "temperature": 0.5,
        "max_tokens": 500,
        "top_p": 0.9,
    }

    # Create model with no tool calls
    model = ModelSettingsTrackingModel(tool_calls=[[]])

    # Create agent with model_settings
    agent = create_agent(
        model=model,
        tools=[sample_tool],
        model_settings=model_settings,
    )

    # Invoke the agent asynchronously
    result = await agent.ainvoke({"messages": [HumanMessage("Async test message")]})

    # Verify all model_settings were passed to bind_tools
    assert "temperature" in bind_kwargs_captured
    assert bind_kwargs_captured["temperature"] == 0.5
    assert "max_tokens" in bind_kwargs_captured
    assert bind_kwargs_captured["max_tokens"] == 500
    assert "top_p" in bind_kwargs_captured
    assert bind_kwargs_captured["top_p"] == 0.9

    # Verify the agent executed successfully
    assert "messages" in result
    assert len(result["messages"]) > 0


def test_create_agent_with_empty_model_settings() -> None:
    """Test that create_agent works with empty model_settings dict."""
    # Track what kwargs were passed to bind_tools
    bind_kwargs_captured = {}

    class ModelSettingsTrackingModel(FakeToolCallingModel):
        """Model that captures kwargs passed to bind_tools."""

        def bind_tools(self, tools, **kwargs):
            bind_kwargs_captured.update(kwargs)
            return super().bind_tools(tools, **kwargs)

    def test_tool(value: str) -> str:
        """A test tool."""
        return value

    model = ModelSettingsTrackingModel(tool_calls=[[]])

    # Create agent with empty model_settings
    agent = create_agent(
        model=model,
        tools=[test_tool],
        model_settings={},
    )

    # Invoke the agent
    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Verify the agent executed successfully without any custom model settings
    assert "messages" in result
    # Should not have temperature or max_tokens since we passed empty dict
    assert "temperature" not in bind_kwargs_captured
    assert "max_tokens" not in bind_kwargs_captured


def test_create_agent_without_model_settings() -> None:
    """Test that create_agent works when model_settings is not provided (defaults to None)."""
    # Track what kwargs were passed to bind_tools
    bind_kwargs_captured = {}

    class ModelSettingsTrackingModel(FakeToolCallingModel):
        """Model that captures kwargs passed to bind_tools."""

        def bind_tools(self, tools, **kwargs):
            bind_kwargs_captured.update(kwargs)
            return super().bind_tools(tools, **kwargs)

    def test_tool(data: int) -> int:
        """A test tool."""
        return data + 1

    model = ModelSettingsTrackingModel(tool_calls=[[]])

    # Create agent without model_settings parameter
    agent = create_agent(
        model=model,
        tools=[test_tool],
    )

    # Invoke the agent
    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Verify the agent executed successfully
    assert "messages" in result
    # Should not have any custom model settings
    assert "temperature" not in bind_kwargs_captured
    assert "max_tokens" not in bind_kwargs_captured


def test_create_agent_model_settings_with_no_tools() -> None:
    """Test that model_settings work even when no tools are provided.

    When there are no tools, the model is bound without tools using bind()
    instead of bind_tools(). The model_settings should still be passed through.
    """
    # Track what kwargs were passed to bind
    bind_kwargs_captured = {}

    class ModelSettingsTrackingModel(FakeToolCallingModel):
        """Model that captures kwargs passed to bind."""

        def bind(self, **kwargs):
            bind_kwargs_captured.update(kwargs)
            return super().bind(**kwargs)

    model_settings = {"temperature": 0.3, "max_tokens": 100}

    model = ModelSettingsTrackingModel(tool_calls=[[]])

    # Create agent without tools but with model_settings
    agent = create_agent(
        model=model,
        tools=None,
        model_settings=model_settings,
    )

    # Invoke the agent
    result = agent.invoke({"messages": [HumanMessage("Test without tools")]})

    # Verify model_settings were passed to bind()
    assert "temperature" in bind_kwargs_captured
    assert bind_kwargs_captured["temperature"] == 0.3
    assert "max_tokens" in bind_kwargs_captured
    assert bind_kwargs_captured["max_tokens"] == 100

    # Verify the agent executed successfully
    assert "messages" in result
    assert len(result["messages"]) > 0


async def test_create_agent_model_settings_with_tool_calls() -> None:
    """Test that model_settings are applied when the model makes tool calls.

    Verifies that model_settings persist across tool-calling rounds in the
    agent loop and that the model behavior is controlled by these settings.
    """
    # Track how many times bind_tools was called and with what kwargs
    bind_calls = []

    class ModelSettingsTrackingModel(FakeToolCallingModel):
        """Model that tracks all bind_tools calls."""

        def bind_tools(self, tools, **kwargs):
            bind_calls.append(kwargs.copy())
            return super().bind_tools(tools, **kwargs)

    def calculator(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    model_settings = {"temperature": 0.1, "max_tokens": 200}

    # Create model that makes one tool call, then returns a response
    model = ModelSettingsTrackingModel(
        tool_calls=[
            [{"name": "calculator", "args": {"x": 5, "y": 3}, "id": "call_1"}],
            [],  # No tool calls on second iteration
        ]
    )

    # Create agent with model_settings
    agent = create_agent(
        model=model,
        tools=[calculator],
        model_settings=model_settings,
    )

    # Invoke the agent
    result = await agent.ainvoke({"messages": [HumanMessage("Calculate 5 + 3")]})

    # Verify the agent executed successfully
    assert "messages" in result

    # Verify model_settings were applied in bind_tools call
    assert len(bind_calls) > 0
    for call_kwargs in bind_calls:
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.1
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 200
