"""Tests for AgentRuntime access via wrap_model_call middleware."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware.types import ModelRequest
from langchain.tools import ToolRuntime

from .model import FakeToolCallingModel


@pytest.fixture
def fake_chat_model():
    """Fixture providing a fake chat model for testing."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="test response")]))


def test_agent_name_accessible_in_middleware(fake_chat_model):
    """Test that agent name can be accessed via middleware."""
    captured_agent_name = None

    @wrap_model_call
    def capture_agent_name(request: ModelRequest, handler):
        nonlocal captured_agent_name
        captured_agent_name = request.runtime.agent_name
        return handler(request)

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[capture_agent_name],
        name="TestAgent",
    )

    agent.invoke({"messages": [HumanMessage("Hello")]})
    assert captured_agent_name == "TestAgent"


def test_nested_agent_name_accessible_in_tool():
    """Test that nested agent's name is accessible when agent is used in a tool."""
    # Track which agent names were captured
    captured_agent_names = []

    @wrap_model_call
    def capture_agent_name(request: ModelRequest, handler):
        captured_agent_names.append(request.runtime.agent_name)
        return handler(request)

    # Create a nested agent that will be called from within a tool
    nested_agent = create_agent(
        FakeToolCallingModel(),
        tools=[],
        middleware=[capture_agent_name],
        name="NestedAgent",
    )

    # Create a tool that invokes the nested agent
    @tool
    def call_nested_agent(query: str, runtime: ToolRuntime) -> str:
        """Tool that calls a nested agent."""
        result = nested_agent.invoke({"messages": [HumanMessage(query)]})
        return result["messages"][-1].content

    # Create outer agent that uses the tool
    outer_agent = create_agent(
        FakeToolCallingModel(
            tool_calls=[
                [{"name": "call_nested_agent", "args": {"query": "test"}, "id": "1"}],
                [],
            ]
        ),
        tools=[call_nested_agent],
        middleware=[capture_agent_name],
        name="OuterAgent",
    )

    # Invoke the outer agent, which should call the tool, which calls the nested agent
    outer_agent.invoke({"messages": [HumanMessage("Hello")]})

    # Both agents should have captured their names
    assert "OuterAgent" in captured_agent_names
    assert "NestedAgent" in captured_agent_names


async def test_agent_name_accessible_in_async_middleware():
    """Test that agent name can be accessed in async middleware."""
    captured_agent_name = None

    @wrap_model_call
    async def capture_agent_name_async(request: ModelRequest, handler):
        nonlocal captured_agent_name
        captured_agent_name = request.runtime.agent_name
        return await handler(request)

    fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="async response")]))
    agent = create_agent(
        fake_model,
        tools=[],
        middleware=[capture_agent_name_async],
        name="AsyncAgent",
    )

    await agent.ainvoke({"messages": [HumanMessage("Hello")]})
    assert captured_agent_name == "AsyncAgent"
