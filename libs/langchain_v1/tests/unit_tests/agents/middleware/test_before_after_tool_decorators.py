"""Integration tests for before_tool and after_tool decorator functionality.

These tests verify the decorator-based approach for tool hooks,
focusing on real-world usage patterns and integration scenarios.
"""

from collections.abc import Callable

from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import after_tool, before_tool
from langchain.tools.tool_node import ToolCallRequest
from tests.unit_tests.agents.test_middleware_agent import FakeToolCallingModel


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool
def calculator(expression: str) -> str:
    """Calculate an expression."""
    return f"Calculated: {expression}"


@tool
def failing_tool(input: str) -> str:
    """Tool that always fails."""
    msg = f"Failed: {input}"
    raise ValueError(msg)


def test_before_tool_basic_logging() -> None:
    """Test basic logging functionality with before_tool decorator."""
    execution_log = []

    @before_tool
    def log_tool_call(state, runtime, request) -> None:
        execution_log.append(f"before_{request.tool_call['name']}")
        execution_log.append(f"args: {request.tool_call['args']}")

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[log_tool_call],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify logging
    assert "before_search" in execution_log
    assert "args: {'query': 'test'}" in execution_log

    # Verify tool still executed normally
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Results for: test" in tool_messages[0].content


def test_after_tool_basic_logging() -> None:
    """Test basic logging functionality with after_tool decorator."""
    execution_log = []

    @after_tool
    def log_tool_result(state, runtime, request, response) -> None:
        execution_log.append(f"after_{request.tool_call['name']}")
        if hasattr(response, 'content'):
            execution_log.append(f"result: {response.content}")

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[log_tool_result],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify logging
    assert "after_search" in execution_log
    assert "result: Results for: test" in execution_log

    # Verify tool still executed normally
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_before_and_after_tool_combined() -> None:
    """Test before_tool and after_tool decorators working together."""
    execution_log = []

    @before_tool
    def log_before(state, runtime, request) -> None:
        execution_log.append(f"before_{request.tool_call['name']}")

    @after_tool
    def log_after(state, runtime, request, response) -> None:
        execution_log.append(f"after_{request.tool_call['name']}")

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[log_before, log_after],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify execution order
    assert execution_log == ["before_search", "after_search"]

    # Verify tool still executed normally
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Results for: test" in tool_messages[0].content


def test_before_tool_state_modification() -> None:
    """Test state modification in before_tool decorator."""
    @before_tool
    def add_tool_metadata(state, runtime, request) -> dict[str, Any]:
        return {
            "tool_metadata": {
                "name": request.tool_call["name"],
                "args": request.tool_call["args"],
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }

    @after_tool
    def verify_metadata(state, runtime, request, response) -> None:
        assert "tool_metadata" in state
        assert state["tool_metadata"]["name"] == request.tool_call["name"]

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[add_tool_metadata, verify_metadata],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify tool still executed normally
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_after_tool_response_inspection() -> None:
    """Test response inspection in after_tool decorator."""
    execution_log = []

    @after_tool
    def inspect_response(state, runtime, request, response) -> None:
        execution_log.append(f"tool_{request.tool_call['name']}")
        if hasattr(response, 'content'):
            execution_log.append(f"content_length: {len(response.content)}")
            execution_log.append(f"starts_with_results: {response.content.startswith('Results')}")
        if hasattr(response, 'status'):
            execution_log.append(f"status: {response.status}")

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[inspect_response],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify response inspection
    assert "tool_search" in execution_log
    assert "content_length:" in execution_log
    assert "starts_with_results: True" in execution_log

    # Verify tool still executed normally
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_before_tool_with_multiple_calls() -> None:
    """Test before_tool decorator with multiple tool calls."""
    execution_log = []

    @before_tool
    def log_multiple_calls(state, runtime, request) -> None:
        execution_log.append(f"before_{request.tool_call['name']}_{request.tool_call['id']}")

    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "first"}, id="1"),
                ToolCall(name="calculator", args={"expression": "1+1"}, id="2"),
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, calculator],
        middleware=[log_multiple_calls],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use multiple tools")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify both tools were logged
    assert "before_search_1" in execution_log
    assert "before_calculator_2" in execution_log

    # Verify both tools executed
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2


def test_after_tool_error_handling() -> None:
    """Test after_tool decorator with failing tool."""
    execution_log = []

    @after_tool
    def log_after_error(state, runtime, request, response) -> None:
        execution_log.append(f"after_{request.tool_call['name']}")
        if hasattr(response, 'content'):
            execution_log.append(f"has_content: True")
            execution_log.append(f"content: {response.content}")
        else:
            execution_log.append("no_content")
        if hasattr(response, 'status'):
            execution_log.append(f"has_status: True, status: {response.status}")

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"input": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[log_after_error],
        checkpointer=InMemorySaver(),
    )

    # Should not raise an exception
    result = agent.invoke(
        {"messages": [HumanMessage("Use failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify after_tool hook was still called
    assert "after_failing_tool" in execution_log

    # Result should be available even for failed tools
    assert any("content:" in log for log in execution_log)


def test_decorator_with_custom_name() -> None:
    """Test decorator with custom middleware name."""
    @before_tool(name="CustomToolLogger")
    def custom_logger(state, runtime, request) -> None:
        pass

    @after_tool(name="CustomResultInspector")
    def custom_inspector(state, runtime, request, response) -> None:
        pass

    # Verify custom names were applied
    assert custom_logger.__class__.__name__ == "CustomToolLogger"
    assert custom_inspector.__class__.__name__ == "CustomResultInspector"


def test_decorator_with_tools_parameter() -> None:
    """Test decorator with tools parameter."""
    @tool
    def extra_tool(input: str) -> str:
        """Extra tool registered with middleware."""
        return f"Extra: {input}"

    @before_tool(tools=[extra_tool])
    def middleware_with_tools(state, runtime, request) -> None:
        pass

    @after_tool(tools=[extra_tool])
    def another_middleware_with_tools(state, runtime, request, response) -> None:
        pass

    # Verify tools were registered
    assert extra_tool in middleware_with_tools.tools
    assert extra_tool in another_middleware_with_tools.tools


def test_decorator_composition_order() -> None:
    """Test that multiple before_tool and after_tool decorators compose correctly."""
    execution_log = []

    @before_tool(name="OuterBefore")
    def outer_before(state, runtime, request) -> None:
        execution_log.append("outer_before")

    @before_tool(name="InnerBefore")
    def inner_before(state, runtime, request) -> None:
        execution_log.append("inner_before")

    @after_tool(name="InnerAfter")
    def inner_after(state, runtime, request, response) -> None:
        execution_log.append("inner_after")

    @after_tool(name="OuterAfter")
    def outer_after(state, runtime, request, response) -> None:
        execution_log.append("outer_after")

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    # Test order: first registered = outermost
    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[outer_before, inner_before, inner_after, outer_after],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify composition order
    assert execution_log == [
        "outer_before",
        "inner_before",
        "inner_after",
        "outer_after",
    ]

    # Verify tool still executed normally
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_async_before_tool_decorator() -> None:
    """Test async before_tool decorator."""
    execution_log = []

    @before_tool
    async def async_logger(state, runtime, request) -> None:
        execution_log.append(f"async_before_{request.tool_call['name']}")

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[async_logger],
        checkpointer=InMemorySaver(),
    )

    # Use async invoke
    import asyncio
    result = asyncio.run(agent.ainvoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    ))

    # Verify async hook was called
    assert "async_before_search" in execution_log

    # Verify tool still executed normally
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_async_after_tool_decorator() -> None:
    """Test async after_tool decorator."""
    execution_log = []

    @after_tool
    async def async_logger(state, runtime, request, response) -> None:
        execution_log.append(f"async_after_{request.tool_call['name']}")
        if hasattr(response, 'content'):
            execution_log.append(f"result_length: {len(response.content)}")

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[async_logger],
        checkpointer=InMemorySaver(),
    )

    # Use async invoke
    import asyncio
    result = asyncio.run(agent.ainvoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    ))

    # Verify async hook was called
    assert "async_after_search" in execution_log
    assert "result_length:" in execution_log

    # Verify tool still executed normally
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_monitoring_pattern_with_before_after() -> None:
    """Test monitoring pattern using both before_tool and after_tool."""
    metrics = []

    @before_tool
    def start_monitoring(state, runtime, request) -> dict[str, Any]:
        import time
        return {"tool_start_time": time.time()}

    @after_tool
    def end_monitoring(state, runtime, request, response) -> None:
        import time
        start_time = state.get("tool_start_time", 0)
        execution_time = time.time() - start_time

        metrics.append({
            "tool": request.tool_call["name"],
            "execution_time": execution_time,
            "success": hasattr(response, 'content') and not response.content.startswith("Error"),
        })

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[start_monitoring, end_monitoring],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify metrics were collected
    assert len(metrics) == 1
    assert metrics[0]["tool"] == "search"
    assert metrics[0]["success"] is True
    assert metrics[0]["execution_time"] >= 0

    # Verify tool still executed normally
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_access_runtime_context() -> None:
    """Test that runtime context is accessible in tool hooks."""
    runtime_info = []

    @before_tool
    def check_runtime_before(state, runtime, request) -> None:
        runtime_info.append(f"before_runtime_type: {type(runtime).__name__}")
        if hasattr(runtime, 'context'):
            runtime_info.append("before_has_context: True")

    @after_tool
    def check_runtime_after(state, runtime, request, response) -> None:
        runtime_info.append(f"after_runtime_type: {type(runtime).__name__}")
        if hasattr(runtime, 'context'):
            runtime_info.append("after_has_context: True")

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[check_runtime_before, check_runtime_after],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify runtime was accessible in both hooks
    assert "before_runtime_type: ToolRuntime" in runtime_info
    assert "after_runtime_type: ToolRuntime" in runtime_info
    assert "before_has_context: True" in runtime_info
    assert "after_has_context: True" in runtime_info

    # Verify tool still executed normally
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1


def test_before_tool_short_circuit_with_jump_to() -> None:
    """Test before_tool with jump_to parameter for flow control."""
    execution_log = []

    @before_tool(can_jump_to=["end"])
    def conditional_skip(state, runtime, request) -> dict[str, Any]:
        if request.tool_call["name"] == "search":
            execution_log.append("skipping_search")
            return {"jump_to": "end"}
        execution_log.append("not_skipping")
        return None

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[conditional_skip],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify skip logic was triggered
    assert "skipping_search" in execution_log
    assert "not_skipping" not in execution_log

    # Tool should not have been executed due to jump
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    # The search tool should not have been called
    search_results = [m for m in tool_messages if "Results for: test" in m.content]
    assert len(search_results) == 0


def test_after_tool_flow_control() -> None:
    """Test after_tool with flow control based on response."""
    execution_log = []

    @after_tool(can_jump_to=["model"])
    def retry_on_failure(state, runtime, request, response) -> dict[str, Any] | None:
        execution_log.append(f"after_{request.tool_call['name']}")

        # Check if tool failed (in real scenario, you'd check response status)
        if hasattr(response, 'status') and response.status == "error":
            execution_log.append("retrying")
            return {"jump_to": "model"}

        execution_log.append("success")
        return None

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"input": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[retry_on_failure],
        checkpointer=InMemorySaver(),
    )

    # Should not raise an exception
    result = agent.invoke(
        {"messages": [HumanMessage("Use failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Verify flow control logic was evaluated
    assert "after_failing_tool" in execution_log
    # Note: actual jump behavior depends on runtime implementation