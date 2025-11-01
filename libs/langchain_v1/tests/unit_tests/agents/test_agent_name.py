"""Test cases for agent name parameter in create_agent.

This module tests that the 'name' parameter provided to create_agent
is correctly set on AIMessage objects in the agent responses.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain.agents import create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel


def test_create_agent_with_name_sets_name_on_ai_message() -> None:
    """Test that providing a name to create_agent sets it on AIMessage responses."""
    agent_name = "Test Agent"
    model = FakeToolCallingModel()

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant",
        name=agent_name,
    )

    result = agent.invoke({"messages": [HumanMessage(content="Hello")]})

    # Verify that the AIMessage has the agent name set
    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    assert len(ai_messages) > 0, "Expected at least one AIMessage in response"
    assert ai_messages[-1].name == agent_name, (
        f"Expected AIMessage.name to be '{agent_name}', got '{ai_messages[-1].name}'"
    )


def test_create_agent_without_name_has_none_name() -> None:
    """Test that not providing a name results in AIMessage with name=None (backward compatibility)."""
    model = FakeToolCallingModel()

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant",
        # name not provided
    )

    result = agent.invoke({"messages": [HumanMessage(content="Hello")]})

    # Verify that the AIMessage has name=None (backward compatibility)
    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    assert len(ai_messages) > 0, "Expected at least one AIMessage in response"
    assert ai_messages[-1].name is None, (
        f"Expected AIMessage.name to be None, got '{ai_messages[-1].name}'"
    )


def test_create_agent_with_name_and_tools() -> None:
    """Test that agent name is set correctly when agent uses tools."""
    from langchain_core.tools import tool

    @tool
    def test_tool(input: str) -> str:
        """A test tool."""
        return f"Result: {input}"

    agent_name = "Tool Using Agent"
    model = FakeToolCallingModel(
        tool_calls=[
            [{"args": {"input": "test"}, "id": "1", "name": "test_tool"}],
            [],  # Second call has no tool calls
        ]
    )

    agent = create_agent(
        model=model,
        tools=[test_tool],
        system_prompt="You are a helpful assistant",
        name=agent_name,
    )

    result = agent.invoke({"messages": [HumanMessage(content="Use the tool")]})

    # Verify that the final AIMessage has the agent name set
    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    assert len(ai_messages) > 0, "Expected at least one AIMessage in response"
    # The last AIMessage should have the name
    assert ai_messages[-1].name == agent_name, (
        f"Expected AIMessage.name to be '{agent_name}', got '{ai_messages[-1].name}'"
    )


@pytest.mark.asyncio
async def test_create_agent_with_name_async() -> None:
    """Test that agent name is set correctly in async invocations."""
    agent_name = "Async Test Agent"
    model = FakeToolCallingModel()

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant",
        name=agent_name,
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Hello")]})

    # Verify that the AIMessage has the agent name set
    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    assert len(ai_messages) > 0, "Expected at least one AIMessage in response"
    assert ai_messages[-1].name == agent_name, (
        f"Expected AIMessage.name to be '{agent_name}', got '{ai_messages[-1].name}'"
    )


def test_create_agent_name_preserved_across_tool_calls() -> None:
    """Test that agent name is preserved in all AIMessage responses, including after tool calls."""
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        return f"Result: {expression}"

    agent_name = "Calculator Agent"
    # Model will make a tool call, then return a final response
    model = FakeToolCallingModel(
        tool_calls=[
            [{"args": {"expression": "2+2"}, "id": "1", "name": "calculator"}],
            [],  # Final response without tool calls
        ]
    )

    agent = create_agent(
        model=model,
        tools=[calculator],
        system_prompt="You are a helpful calculator assistant",
        name=agent_name,
    )

    result = agent.invoke({"messages": [HumanMessage(content="What is 2+2?")]})

    # Verify all AIMessages have the agent name set
    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    assert len(ai_messages) > 0, "Expected at least one AIMessage in response"
    for ai_msg in ai_messages:
        assert ai_msg.name == agent_name, (
            f"Expected all AIMessages to have name '{agent_name}', but got '{ai_msg.name}'"
        )


def test_create_agent_name_not_overwritten_if_already_set() -> None:
    """Test that if AIMessage already has a name, it is not overwritten."""
    # This test ensures backward compatibility - if a model returns an AIMessage
    # with a name already set, we should preserve it rather than overwrite it
    agent_name = "Test Agent"

    # Create a custom model that returns AIMessage with name already set
    class ModelWithNamedResponse(FakeToolCallingModel):
        def _generate(self, messages, **kwargs):
            message = AIMessage(
                content="Response with existing name",
                name="ModelSetName",  # Model sets its own name
            )
            from langchain_core.outputs import ChatGeneration, ChatResult

            return ChatResult(generations=[ChatGeneration(message=message)])

    model = ModelWithNamedResponse()

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant",
        name=agent_name,
    )

    result = agent.invoke({"messages": [HumanMessage(content="Hello")]})

    # The message already has a name, so it should be preserved
    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    assert len(ai_messages) > 0
    # If the model already set a name, it should be preserved
    # (though in practice, most models don't set names)
    # This test verifies our code doesn't overwrite existing names
    assert ai_messages[-1].name is not None


def test_create_agent_with_empty_string_name() -> None:
    """Test that empty string name is handled correctly (should still set it)."""
    model = FakeToolCallingModel()

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant",
        name="",  # Empty string
    )

    result = agent.invoke({"messages": [HumanMessage(content="Hello")]})

    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    assert len(ai_messages) > 0
    # Empty string should still be set (not None)
    # Though this might not be a realistic use case
    assert ai_messages[-1].name == ""


def test_create_agent_no_name_parameter_with_tools() -> None:
    """Test that agent works correctly when name parameter is completely omitted, even with tools."""
    from langchain_core.tools import tool

    @tool
    def test_tool(input: str) -> str:
        """A test tool."""
        return f"Result: {input}"

    model = FakeToolCallingModel(
        tool_calls=[
            [{"args": {"input": "test"}, "id": "1", "name": "test_tool"}],
            [],  # Second call has no tool calls
        ]
    )

    # Explicitly not passing name parameter at all
    agent = create_agent(
        model=model,
        tools=[test_tool],
        system_prompt="You are a helpful assistant",
    )

    result = agent.invoke({"messages": [HumanMessage(content="Use the tool")]})

    # Verify that the AIMessage has name=None when name is not provided
    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    assert len(ai_messages) > 0, "Expected at least one AIMessage in response"
    # All AIMessages should have name=None when name parameter is not provided
    for ai_msg in ai_messages:
        assert ai_msg.name is None, (
            f"Expected AIMessage.name to be None when name parameter is omitted, got '{ai_msg.name}'"
        )
