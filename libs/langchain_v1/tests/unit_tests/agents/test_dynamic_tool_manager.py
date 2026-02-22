"""Tests for DynamicToolManager — dynamic tool addition/removal after agent creation.

Verifies the `add_tool`, `remove_tool`, `add_tools`, `remove_tools` API
exposed on the compiled agent graph via the built-in DynamicToolManager middleware.
"""

import asyncio
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.factory import create_agent
from langchain.agents.tool_manager import DynamicToolManager
from tests.unit_tests.agents.model import FakeToolCallingModel

# -- Tool fixtures --


@tool
def static_tool(value: str) -> str:
    """A static tool that is always available."""
    return f"Static result: {value}"


@tool
def dynamic_tool_a(value: str) -> str:
    """A dynamically registered tool."""
    return f"Dynamic A: {value}"


@tool
def dynamic_tool_b(x: int, y: int) -> str:
    """Another dynamically registered tool."""
    return f"Sum: {x + y}"


# -- Helpers --


def get_tool_messages(result: dict[str, Any]) -> list[ToolMessage]:
    """Extract ToolMessage objects from agent result."""
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


async def invoke_agent(agent: Any, message: str, *, use_async: bool) -> dict[str, Any]:
    """Invoke agent synchronously or asynchronously."""
    input_data = {"messages": [HumanMessage(message)]}
    config = {"configurable": {"thread_id": "test"}}
    if use_async:
        return await agent.ainvoke(input_data, config)
    return await asyncio.to_thread(agent.invoke, input_data, config)


# -- Tests --


@pytest.mark.parametrize("use_async", [False, True])
async def test_add_tool_after_creation(*, use_async: bool) -> None:
    """Adding a tool after create_agent() should make it available."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="dynamic_tool_a", args={"value": "hello"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        checkpointer=InMemorySaver(),
    )

    # Add tool after creation
    agent.add_tool(dynamic_tool_a)

    result = await invoke_agent(agent, "Use dynamic tool", use_async=use_async)

    tool_messages = get_tool_messages(result)
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "dynamic_tool_a"
    assert "Dynamic A: hello" in tool_messages[0].content


@pytest.mark.parametrize("use_async", [False, True])
async def test_remove_tool_after_creation(*, use_async: bool) -> None:
    """Removing a dynamically added tool should make it unavailable."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="static_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        checkpointer=InMemorySaver(),
    )

    # Add then remove
    agent.add_tool(dynamic_tool_a)
    agent.remove_tool("dynamic_tool_a")

    # Agent should still work with static tool
    result = await invoke_agent(agent, "Use static tool", use_async=use_async)

    tool_messages = get_tool_messages(result)
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "static_tool"
    assert "Static result: test" in tool_messages[0].content


@pytest.mark.parametrize("use_async", [False, True])
async def test_add_tools_batch(*, use_async: bool) -> None:
    """add_tools() should add multiple tools at once."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="dynamic_tool_a", args={"value": "first"}, id="1"),
                ToolCall(name="dynamic_tool_b", args={"x": 3, "y": 7}, id="2"),
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        checkpointer=InMemorySaver(),
    )

    agent.add_tools([dynamic_tool_a, dynamic_tool_b])

    result = await invoke_agent(agent, "Use both dynamic tools", use_async=use_async)

    tool_messages = get_tool_messages(result)
    assert len(tool_messages) == 2

    results_by_name = {m.name: m.content for m in tool_messages}
    assert "Dynamic A: first" in results_by_name["dynamic_tool_a"]
    assert "Sum: 10" in results_by_name["dynamic_tool_b"]


@pytest.mark.parametrize("use_async", [False, True])
async def test_remove_tools_batch(*, use_async: bool) -> None:
    """remove_tools() should remove multiple tools at once."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="static_tool", args={"value": "only"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        checkpointer=InMemorySaver(),
    )

    agent.add_tools([dynamic_tool_a, dynamic_tool_b])
    agent.remove_tools(["dynamic_tool_a", "dynamic_tool_b"])

    result = await invoke_agent(agent, "Use static tool", use_async=use_async)

    tool_messages = get_tool_messages(result)
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "static_tool"


def test_remove_nonexistent_tool_raises() -> None:
    """remove_tool() with unknown name should raise ValueError."""
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        tools=[static_tool],
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(ValueError, match="not found in dynamic tools"):
        agent.remove_tool("nonexistent_tool")


@pytest.mark.parametrize("use_async", [False, True])
async def test_add_tool_with_callable(*, use_async: bool) -> None:
    """add_tool() should accept a plain callable and convert it."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="greet", args={"name": "World"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        checkpointer=InMemorySaver(),
    )

    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    agent.add_tool(greet)

    result = await invoke_agent(agent, "Greet someone", use_async=use_async)

    tool_messages = get_tool_messages(result)
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "greet"
    assert "Hello, World!" in tool_messages[0].content


@pytest.mark.parametrize("use_async", [False, True])
async def test_dynamic_and_static_tools_coexist(*, use_async: bool) -> None:
    """Static and dynamic tools should both work in the same invocation."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="static_tool", args={"value": "static"}, id="1"),
                ToolCall(
                    name="dynamic_tool_a",
                    args={"value": "dynamic"},
                    id="2",
                ),
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[static_tool],
        checkpointer=InMemorySaver(),
    )

    agent.add_tool(dynamic_tool_a)

    result = await invoke_agent(agent, "Use both tools", use_async=use_async)

    tool_messages = get_tool_messages(result)
    assert len(tool_messages) == 2

    results_by_name = {m.name: m.content for m in tool_messages}
    assert "Static result: static" in results_by_name["static_tool"]
    assert "Dynamic A: dynamic" in results_by_name["dynamic_tool_a"]


def test_add_tool_without_initial_tools_raises() -> None:
    """add_tool() on agent created with no tools should raise RuntimeError."""
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        tools=[],
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(RuntimeError, match="Cannot add dynamic tools"):
        agent.add_tool(dynamic_tool_a)


def test_dynamic_tools_property() -> None:
    """DynamicToolManager.dynamic_tools should reflect current state."""
    manager = DynamicToolManager()
    # Provide a mock tool node so add_tool/remove_tool work
    manager.set_tool_node(_make_mock_tool_node())
    assert manager.dynamic_tools == []

    manager.add_tool(dynamic_tool_a)
    assert len(manager.dynamic_tools) == 1
    assert manager.dynamic_tools[0].name == "dynamic_tool_a"

    manager.add_tool(dynamic_tool_b)
    assert len(manager.dynamic_tools) == 2

    manager.remove_tool("dynamic_tool_a")
    assert len(manager.dynamic_tools) == 1
    assert manager.dynamic_tools[0].name == "dynamic_tool_b"


def test_add_tool_replaces_existing() -> None:
    """Adding a tool with the same name should replace it."""
    manager = DynamicToolManager()
    manager.set_tool_node(_make_mock_tool_node())

    manager.add_tool(dynamic_tool_a)
    assert len(manager.dynamic_tools) == 1

    # Add a different tool with the same name
    replacement = StructuredTool.from_function(
        func=lambda value: f"Replaced: {value}",
        name="dynamic_tool_a",
        description="Replacement tool.",
    )

    manager.add_tool(replacement)
    assert len(manager.dynamic_tools) == 1


def test_dynamic_tool_manager_is_base_tool_aware() -> None:
    """add_tool should handle both BaseTool and callable."""
    manager = DynamicToolManager()
    manager.set_tool_node(_make_mock_tool_node())

    # BaseTool instance
    manager.add_tool(dynamic_tool_a)
    assert isinstance(manager.dynamic_tools[0], BaseTool)

    # Plain callable
    def my_func(x: int) -> str:
        """Do something."""
        return str(x)

    manager.add_tool(my_func)
    assert len(manager.dynamic_tools) == 2
    assert all(isinstance(t, BaseTool) for t in manager.dynamic_tools)


# -- Helpers for standalone DynamicToolManager tests --


class _MockToolNode:
    """Minimal mock that provides tools_by_name for standalone tests."""

    def __init__(self) -> None:
        self.tools_by_name: dict[str, BaseTool] = {}


def _make_mock_tool_node() -> Any:
    """Create a mock ToolNode for standalone DynamicToolManager tests."""
    return _MockToolNode()
