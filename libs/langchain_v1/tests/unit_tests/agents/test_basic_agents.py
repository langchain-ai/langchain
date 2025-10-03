"""Tests for basic agent functionality."""

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from typing import Any
from langchain_core.tools import tool as dec_tool, InjectedToolCallId, ToolException
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import (
    InjectedState,
    InjectedStore,
    ToolNode,
)
from langchain.tools.tool_node import (
    _get_state_args,
    _infer_handled_types,
)

from .any_str import AnyStr
from .messages import _AnyIdHumanMessage, _AnyIdToolMessage
from .model import FakeToolCallingModel

pytestmark = pytest.mark.anyio


def test_no_prompt(sync_checkpointer: BaseCheckpointSaver) -> None:
    """Test agent with no system prompt."""
    model = FakeToolCallingModel()

    agent = create_agent(
        model,
        [],
        checkpointer=sync_checkpointer,
    )
    inputs = [HumanMessage("hi?")]
    thread = {"configurable": {"thread_id": "123"}}
    response = agent.invoke({"messages": inputs}, thread, debug=True)
    expected_response = {"messages": [*inputs, AIMessage(content="hi?", id="0")]}
    assert response == expected_response

    saved = sync_checkpointer.get_tuple(thread)
    assert saved is not None
    checkpoint_values = saved.checkpoint["channel_values"]
    assert checkpoint_values["messages"] == [
        _AnyIdHumanMessage(content="hi?"),
        AIMessage(content="hi?", id="0"),
    ]
    assert checkpoint_values["thread_model_call_count"] == 1
    assert saved.metadata == {
        "parents": {},
        "source": "loop",
        "step": 1,
    }
    assert saved.pending_writes == []


async def test_no_prompt_async(async_checkpointer: BaseCheckpointSaver) -> None:
    """Test agent with no system prompt (async)."""
    model = FakeToolCallingModel()

    agent = create_agent(model, [], checkpointer=async_checkpointer)
    inputs = [HumanMessage("hi?")]
    thread = {"configurable": {"thread_id": "123"}}
    response = await agent.ainvoke({"messages": inputs}, thread, debug=True)
    expected_response = {"messages": [*inputs, AIMessage(content="hi?", id="0")]}
    assert response == expected_response

    saved = await async_checkpointer.aget_tuple(thread)
    assert saved is not None
    checkpoint_values = saved.checkpoint["channel_values"]
    assert checkpoint_values["messages"] == [
        _AnyIdHumanMessage(content="hi?"),
        AIMessage(content="hi?", id="0"),
    ]
    assert checkpoint_values["thread_model_call_count"] == 1
    assert saved.metadata == {
        "parents": {},
        "source": "loop",
        "step": 1,
    }
    assert saved.pending_writes == []


def test_system_message_prompt() -> None:
    """Test agent with system message prompt."""
    system_prompt = "Foo"
    model = FakeToolCallingModel()

    agent = create_agent(model, [], system_prompt=system_prompt)
    inputs = [HumanMessage("hi?")]
    response = agent.invoke({"messages": inputs}, debug=True)
    expected_response = {"messages": [*inputs, AIMessage(content="hi?", id="0")]}
    assert response == expected_response


def test_system_message_prompt_with_tools() -> None:
    """Test agent with system message prompt and tools."""
    system_prompt = "You are a helpful assistant."

    @dec_tool
    def search_tool(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    model = FakeToolCallingModel(
        tool_calls=[[{"args": {"query": "test"}, "id": "1", "name": "search_tool"}], []]
    )

    agent = create_agent(model, [search_tool], system_prompt=system_prompt)
    inputs = [HumanMessage("Search for something")]
    response = agent.invoke({"messages": inputs}, debug=True)

    assert "messages" in response
    messages = response["messages"]
    assert len(messages) >= 2  # Human message + AI message
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)


def test_agent_with_tools() -> None:
    """Test agent with tools."""

    @dec_tool
    def calculator(expression: str) -> str:
        """Calculate a mathematical expression."""
        return f"Result: {expression}"

    @dec_tool
    def search_tool(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    model = FakeToolCallingModel(
        tool_calls=[[{"args": {"expression": "2+2"}, "id": "1", "name": "calculator"}], []]
    )

    agent = create_agent(model, [calculator, search_tool])
    inputs = [HumanMessage("Calculate 2+2")]
    response = agent.invoke({"messages": inputs}, debug=True)

    assert "messages" in response
    messages = response["messages"]
    assert len(messages) >= 3  # Human + AI + Tool message
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)


def test_agent_with_structured_output() -> None:
    """Test agent with structured output."""

    class WeatherResponse(BaseModel):
        temperature: float = Field(description="Temperature in Fahrenheit")
        condition: str = Field(description="Weather condition")

    model = FakeToolCallingModel()

    agent = create_agent(
        model,
        [],
        response_format=ToolStrategy(schema=WeatherResponse),
    )
    inputs = [HumanMessage("What's the weather like?")]
    response = agent.invoke({"messages": inputs}, debug=True)

    assert "messages" in response
    messages = response["messages"]
    assert len(messages) >= 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)


def test_agent_with_injected_tools() -> None:
    """Test agent with tools that use injected state."""

    @dec_tool
    def state_tool(input: str, state: Annotated[dict, InjectedState]) -> str:
        """Tool that uses injected state."""
        return f"State: {state.get('custom_field', 'none')}"

    @dec_tool
    def store_tool(input: str, store: Annotated[BaseStore, InjectedStore]) -> str:
        """Tool that uses injected store."""
        return f"Store: {type(store).__name__}"

    model = FakeToolCallingModel(
        tool_calls=[[{"args": {"input": "test"}, "id": "1", "name": "state_tool"}], []]
    )

    agent = create_agent(
        model,
        [state_tool, store_tool],
        store=InMemoryStore(),
    )
    inputs = [HumanMessage("Use state tool")]
    response = agent.invoke({"messages": inputs, "custom_field": "test_value"}, debug=True)

    assert "messages" in response
    messages = response["messages"]
    assert len(messages) >= 3  # Human + AI + Tool message
    assert isinstance(messages[2], ToolMessage)
    assert "test_value" in messages[2].content


def test_agent_with_tool_exception() -> None:
    """Test agent handling tool exceptions."""

    @dec_tool
    def error_tool(input: str) -> str:
        """Tool that raises an exception."""
        raise ToolException("Tool error occurred")

    model = FakeToolCallingModel(
        tool_calls=[[{"args": {"input": "test"}, "id": "1", "name": "error_tool"}], []]
    )

    agent = create_agent(model, [error_tool])
    inputs = [HumanMessage("Use error tool")]
    response = agent.invoke({"messages": inputs}, debug=True)

    assert "messages" in response
    messages = response["messages"]
    assert len(messages) >= 3  # Human + AI + Tool message
    assert isinstance(messages[2], ToolMessage)
    assert "Tool error occurred" in messages[2].content


def test_agent_with_multiple_tool_calls() -> None:
    """Test agent with multiple tool calls in one response."""

    @dec_tool
    def tool1(input: str) -> str:
        """First tool."""
        return f"Tool1: {input}"

    @dec_tool
    def tool2(input: str) -> str:
        """Second tool."""
        return f"Tool2: {input}"

    model = FakeToolCallingModel(
        tool_calls=[
            [
                {"args": {"input": "test1"}, "id": "1", "name": "tool1"},
                {"args": {"input": "test2"}, "id": "2", "name": "tool2"},
            ],
            [],
        ]
    )

    agent = create_agent(model, [tool1, tool2])
    inputs = [HumanMessage("Use both tools")]
    response = agent.invoke({"messages": inputs}, debug=True)

    assert "messages" in response
    messages = response["messages"]
    assert len(messages) >= 4  # Human + AI + 2 Tool messages
    assert isinstance(messages[1], AIMessage)
    assert len(messages[1].tool_calls) == 2
    assert isinstance(messages[2], ToolMessage)
    assert isinstance(messages[3], ToolMessage)


def test_agent_with_custom_middleware() -> None:
    """Test agent with custom middleware."""

    class CustomMiddleware(AgentMiddleware[AgentState]):
        def before_model(self, state: AgentState, runtime) -> dict[str, Any]:
            return {"custom_field": "middleware_value"}

    model = FakeToolCallingModel()

    agent = create_agent(model, [], middleware=[CustomMiddleware()])
    inputs = [HumanMessage("Hello")]
    response = agent.invoke({"messages": inputs}, debug=True)

    assert "messages" in response
    assert "custom_field" in response
    assert response["custom_field"] == "middleware_value"


def test_agent_with_checkpointer() -> None:
    """Test agent with checkpointer for state persistence."""
    model = FakeToolCallingModel()

    agent = create_agent(model, [], checkpointer=InMemoryStore())
    inputs = [HumanMessage("Hello")]
    thread = {"configurable": {"thread_id": "test_thread"}}

    # First invocation
    response1 = agent.invoke({"messages": inputs}, thread, debug=True)
    assert "messages" in response1

    # Second invocation in same thread
    inputs2 = [HumanMessage("Hello again")]
    response2 = agent.invoke({"messages": inputs2}, thread, debug=True)
    assert "messages" in response2

    # Should have conversation history
    messages = response2["messages"]
    assert len(messages) >= 2  # Should have previous messages


def test_agent_with_store() -> None:
    """Test agent with store for persistent data."""

    @dec_tool
    def store_tool(input: str, store: Annotated[BaseStore, InjectedStore]) -> str:
        """Tool that uses store."""
        store.put("test_key", "test_value")
        return "Stored value"

    model = FakeToolCallingModel(
        tool_calls=[[{"args": {"input": "test"}, "id": "1", "name": "store_tool"}], []]
    )

    store = InMemoryStore()
    agent = create_agent(model, [store_tool], store=store)
    inputs = [HumanMessage("Use store tool")]
    response = agent.invoke({"messages": inputs}, debug=True)

    assert "messages" in response
    # Verify store was used
    stored_value = store.get("test_key")
    assert stored_value == "test_value"


def test_agent_debug_mode() -> None:
    """Test agent in debug mode."""
    model = FakeToolCallingModel()

    agent = create_agent(model, [])
    inputs = [HumanMessage("Hello")]
    response = agent.invoke({"messages": inputs}, debug=True)

    assert "messages" in response
    # Debug mode should provide additional information
    assert isinstance(response, dict)


def test_agent_with_empty_tools() -> None:
    """Test agent with empty tools list."""
    model = FakeToolCallingModel()

    agent = create_agent(model, [])
    inputs = [HumanMessage("Hello")]
    response = agent.invoke({"messages": inputs}, debug=True)

    assert "messages" in response
    messages = response["messages"]
    assert len(messages) >= 2  # Human + AI message
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)


def test_agent_with_none_system_prompt() -> None:
    """Test agent with None system prompt."""
    model = FakeToolCallingModel()

    agent = create_agent(model, [], system_prompt=None)
    inputs = [HumanMessage("Hello")]
    response = agent.invoke({"messages": inputs}, debug=True)

    assert "messages" in response
    messages = response["messages"]
    assert len(messages) >= 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
