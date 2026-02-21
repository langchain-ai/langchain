"""Tests for ToolRetryMiddleware functionality."""

import time
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from langchain.agents.factory import create_agent
from langchain.agents.middleware._retry import calculate_delay
from langchain.agents.middleware.tool_retry import ToolRetryMiddleware
from langchain.agents.middleware.types import wrap_tool_call
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def working_tool(value: str) -> str:
    """Tool that always succeeds."""
    return f"Success: {value}"


@tool
def failing_tool(value: str) -> str:
    """Tool that always fails."""
    msg = f"Failed: {value}"
    raise ValueError(msg)


class TemporaryFailureTool:
    """Tool that fails a certain number of times before succeeding."""

    def __init__(self, fail_count: int):
        """Initialize with the number of times to fail.

        Args:
            fail_count: Number of times to fail before succeeding.
        """
        self.fail_count = fail_count
        self.attempt = 0

    def __call__(self, value: str) -> str:
        """Execute the tool.

        Args:
            value: Input string.

        Returns:
            Success message if attempt >= fail_count.

        Raises:
            ValueError: If attempt < fail_count.
        """
        self.attempt += 1
        if self.attempt <= self.fail_count:
            msg = f"Temporary failure {self.attempt}"
            raise ValueError(msg)
        return f"Success after {self.attempt} attempts: {value}"


def test_tool_retry_initialization_defaults() -> None:
    """Test ToolRetryMiddlewareinitialization with default values."""
    retry = ToolRetryMiddleware()

    assert retry.max_retries == 2
    assert retry._tool_filter is None
    assert retry.tools == []
    assert retry.on_failure == "continue"
    assert retry.backoff_factor == 2.0
    assert retry.initial_delay == 1.0
    assert retry.max_delay == 60.0
    assert retry.jitter is True


def test_tool_retry_initialization_custom() -> None:
    """Test ToolRetryMiddlewareinitialization with custom values."""
    retry = ToolRetryMiddleware(
        max_retries=5,
        tools=["tool1", "tool2"],
        retry_on=(ValueError, RuntimeError),
        on_failure="error",
        backoff_factor=1.5,
        initial_delay=0.5,
        max_delay=30.0,
        jitter=False,
    )

    assert retry.max_retries == 5
    assert retry._tool_filter == ["tool1", "tool2"]
    assert retry.tools == []
    assert retry.retry_on == (ValueError, RuntimeError)
    assert retry.on_failure == "error"
    assert retry.backoff_factor == 1.5
    assert retry.initial_delay == 0.5
    assert retry.max_delay == 30.0
    assert retry.jitter is False


def test_tool_retry_initialization_with_base_tools() -> None:
    """Test ToolRetryMiddleware initialization with BaseTool instances."""
    retry = ToolRetryMiddleware(
        max_retries=3,
        tools=[working_tool, failing_tool],  # Pass BaseTool instances
    )

    assert retry.max_retries == 3
    # Should extract names from BaseTool instances
    assert retry._tool_filter == ["working_tool", "failing_tool"]
    assert retry.tools == []


def test_tool_retry_initialization_with_mixed_tools() -> None:
    """Test ToolRetryMiddleware initialization with mixed tool types."""
    retry = ToolRetryMiddleware(
        max_retries=2,
        tools=[working_tool, "failing_tool"],  # Mix of BaseTool and string
    )

    assert retry.max_retries == 2
    # Should handle both BaseTool instances and strings
    assert retry._tool_filter == ["working_tool", "failing_tool"]
    assert retry.tools == []


def test_tool_retry_invalid_max_retries() -> None:
    """Test ToolRetryMiddlewareraises error for invalid max_retries."""
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        ToolRetryMiddleware(max_retries=-1)


def test_tool_retry_invalid_initial_delay() -> None:
    """Test ToolRetryMiddlewareraises error for invalid initial_delay."""
    with pytest.raises(ValueError, match="initial_delay must be >= 0"):
        ToolRetryMiddleware(initial_delay=-1.0)


def test_tool_retry_invalid_max_delay() -> None:
    """Test ToolRetryMiddlewareraises error for invalid max_delay."""
    with pytest.raises(ValueError, match="max_delay must be >= 0"):
        ToolRetryMiddleware(max_delay=-1.0)


def test_tool_retry_invalid_backoff_factor() -> None:
    """Test ToolRetryMiddlewareraises error for invalid backoff_factor."""
    with pytest.raises(ValueError, match="backoff_factor must be >= 0"):
        ToolRetryMiddleware(backoff_factor=-1.0)


def test_tool_retry_working_tool_no_retry_needed() -> None:
    """Test ToolRetryMiddlewarewith a working tool (no retry needed)."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(max_retries=2, initial_delay=0.01, jitter=False)

    agent = create_agent(
        model=model,
        tools=[working_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use working tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Success: test" in tool_messages[0].content
    assert tool_messages[0].status != "error"


def test_tool_retry_failing_tool_returns_message() -> None:
    """Test ToolRetryMiddlewarewith failing tool returns error message."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    # Should contain error message with tool name and attempts
    assert "failing_tool" in tool_messages[0].content
    assert "3 attempts" in tool_messages[0].content
    assert "ValueError" in tool_messages[0].content
    assert tool_messages[0].status == "error"


def test_tool_retry_failing_tool_raises() -> None:
    """Test ToolRetryMiddlewarewith on_failure='error' re-raises exception."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        on_failure="error",
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    # Should raise the ValueError from the tool
    with pytest.raises(ValueError, match="Failed: test"):
        agent.invoke(
            {"messages": [HumanMessage("Use failing tool")]},
            {"configurable": {"thread_id": "test"}},
        )


def test_tool_retry_custom_failure_formatter() -> None:
    """Test ToolRetryMiddlewarewith custom failure message formatter."""

    def custom_formatter(exc: Exception) -> str:
        return f"Custom error: {type(exc).__name__}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(
        max_retries=1,
        initial_delay=0.01,
        jitter=False,
        on_failure=custom_formatter,
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Custom error: ValueError" in tool_messages[0].content


def test_tool_retry_succeeds_after_retries() -> None:
    """Test ToolRetryMiddlewaresucceeds after temporary failures."""
    temp_fail = TemporaryFailureTool(fail_count=2)

    @tool
    def temp_failing_tool(value: str) -> str:
        """Tool that fails temporarily."""
        return temp_fail(value)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="temp_failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(
        max_retries=3,
        initial_delay=0.01,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[temp_failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use temp failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    # Should succeed on 3rd attempt
    assert "Success after 3 attempts" in tool_messages[0].content
    assert tool_messages[0].status != "error"


def test_tool_retry_specific_tools_only() -> None:
    """Test ToolRetryMiddlewareonly applies to specific tools."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="failing_tool", args={"value": "test1"}, id="1"),
                ToolCall(name="working_tool", args={"value": "test2"}, id="2"),
            ],
            [],
        ]
    )

    # Only retry failing_tool
    retry = ToolRetryMiddleware(
        max_retries=2,
        tools=["failing_tool"],
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool, working_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use both tools")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2

    # failing_tool should have error message
    failing_msg = next(m for m in tool_messages if m.name == "failing_tool")
    assert failing_msg.status == "error"
    assert "3 attempts" in failing_msg.content

    # working_tool should succeed normally (no retry applied)
    working_msg = next(m for m in tool_messages if m.name == "working_tool")
    assert "Success: test2" in working_msg.content
    assert working_msg.status != "error"


def test_tool_retry_specific_tools_with_base_tool() -> None:
    """Test ToolRetryMiddleware accepts BaseTool instances for filtering."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="failing_tool", args={"value": "test1"}, id="1"),
                ToolCall(name="working_tool", args={"value": "test2"}, id="2"),
            ],
            [],
        ]
    )

    # Only retry failing_tool, passed as BaseTool instance
    retry = ToolRetryMiddleware(
        max_retries=2,
        tools=[failing_tool],  # Pass BaseTool instance
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool, working_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use both tools")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2

    # failing_tool should have error message (with retries)
    failing_msg = next(m for m in tool_messages if m.name == "failing_tool")
    assert failing_msg.status == "error"
    assert "3 attempts" in failing_msg.content

    # working_tool should succeed normally (no retry applied)
    working_msg = next(m for m in tool_messages if m.name == "working_tool")
    assert "Success: test2" in working_msg.content
    assert working_msg.status != "error"


def test_tool_retry_specific_exceptions() -> None:
    """Test ToolRetryMiddlewareonly retries specific exception types."""

    @tool
    def value_error_tool(value: str) -> str:
        """Tool that raises ValueError."""
        msg = f"ValueError: {value}"
        raise ValueError(msg)

    @tool
    def runtime_error_tool(value: str) -> str:
        """Tool that raises RuntimeError."""
        msg = f"RuntimeError: {value}"
        raise RuntimeError(msg)

    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="value_error_tool", args={"value": "test1"}, id="1"),
                ToolCall(name="runtime_error_tool", args={"value": "test2"}, id="2"),
            ],
            [],
        ]
    )

    # Only retry ValueError
    retry = ToolRetryMiddleware(
        max_retries=2,
        retry_on=(ValueError,),
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[value_error_tool, runtime_error_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use both tools")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2

    # ValueError should be retried (3 attempts)
    value_error_msg = next(m for m in tool_messages if m.name == "value_error_tool")
    assert "3 attempts" in value_error_msg.content

    # RuntimeError should fail immediately (1 attempt only)
    runtime_error_msg = next(m for m in tool_messages if m.name == "runtime_error_tool")
    assert "1 attempt" in runtime_error_msg.content


def test_tool_retry_custom_exception_filter() -> None:
    """Test ToolRetryMiddlewarewith custom exception filter function."""

    class CustomError(Exception):
        """Custom exception with retry_me attribute."""

        def __init__(self, message: str, *, retry_me: bool):
            """Initialize custom error.

            Args:
                message: Error message.
                retry_me: Whether this error should be retried.
            """
            super().__init__(message)
            self.retry_me = retry_me

    attempt_count = {"value": 0}

    @tool
    def custom_error_tool(val: str) -> str:
        """Tool that raises CustomError."""
        _ = val
        attempt_count["value"] += 1
        if attempt_count["value"] == 1:
            msg = "Retryable error"
            raise CustomError(msg, retry_me=True)
        msg = "Non-retryable error"
        raise CustomError(msg, retry_me=False)

    def should_retry(exc: Exception) -> bool:
        return isinstance(exc, CustomError) and exc.retry_me

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="custom_error_tool", args={"val": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(
        max_retries=3,
        retry_on=should_retry,
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[custom_error_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use custom error tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1

    # Should retry once (attempt 1 with retry_me=True), then fail on attempt 2 (retry_me=False)
    assert attempt_count["value"] == 2
    assert "2 attempts" in tool_messages[0].content


def test_tool_retry_backoff_timing() -> None:
    """Test ToolRetryMiddlewareapplies correct backoff delays."""
    temp_fail = TemporaryFailureTool(fail_count=3)

    @tool
    def temp_failing_tool(value: str) -> str:
        """Tool that fails temporarily."""
        return temp_fail(value)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="temp_failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(
        max_retries=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[temp_failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    start_time = time.time()
    result = agent.invoke(
        {"messages": [HumanMessage("Use temp failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )
    elapsed = time.time() - start_time

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1

    # Expected delays: 0.1 + 0.2 + 0.4 = 0.7 seconds
    # Allow some margin for execution time
    assert elapsed >= 0.6, f"Expected at least 0.6s, got {elapsed}s"


def test_tool_retry_constant_backoff() -> None:
    """Test ToolRetryMiddlewarewith constant backoff (backoff_factor=0)."""
    temp_fail = TemporaryFailureTool(fail_count=2)

    @tool
    def temp_failing_tool(value: str) -> str:
        """Tool that fails temporarily."""
        return temp_fail(value)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="temp_failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(
        max_retries=2,
        initial_delay=0.1,
        backoff_factor=0.0,  # Constant backoff
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[temp_failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    start_time = time.time()
    result = agent.invoke(
        {"messages": [HumanMessage("Use temp failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )
    elapsed = time.time() - start_time

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1

    # Expected delays: 0.1 + 0.1 = 0.2 seconds (constant)
    assert elapsed >= 0.15, f"Expected at least 0.15s, got {elapsed}s"
    assert elapsed < 0.5, f"Expected less than 0.5s (exponential would be longer), got {elapsed}s"


def test_tool_retry_max_delay_cap() -> None:
    """Test calculate_delay caps delay at max_delay."""
    # Test delay calculation with aggressive backoff and max_delay cap
    delay_0 = calculate_delay(
        0,
        backoff_factor=10.0,  # Very aggressive backoff
        initial_delay=1.0,
        max_delay=2.0,  # Cap at 2 seconds
        jitter=False,
    )  # 1.0
    delay_1 = calculate_delay(
        1,
        backoff_factor=10.0,
        initial_delay=1.0,
        max_delay=2.0,
        jitter=False,
    )  # 10.0 -> capped to 2.0
    delay_2 = calculate_delay(
        2,
        backoff_factor=10.0,
        initial_delay=1.0,
        max_delay=2.0,
        jitter=False,
    )  # 100.0 -> capped to 2.0

    assert delay_0 == 1.0
    assert delay_1 == 2.0
    assert delay_2 == 2.0


def test_tool_retry_jitter_variation() -> None:
    """Test calculate_delay adds jitter to delays."""
    # Generate multiple delays and ensure they vary
    delays = [
        calculate_delay(
            0,
            backoff_factor=1.0,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=True,
        )
        for _ in range(10)
    ]

    # All delays should be within ±25% of 1.0 (i.e., between 0.75 and 1.25)
    for delay in delays:
        assert 0.75 <= delay <= 1.25

    # Delays should vary (not all the same)
    assert len(set(delays)) > 1


async def test_tool_retry_async_working_tool() -> None:
    """Test ToolRetryMiddlewarewith async execution and working tool."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(max_retries=2, initial_delay=0.01, jitter=False)

    agent = create_agent(
        model=model,
        tools=[working_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use working tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Success: test" in tool_messages[0].content


async def test_tool_retry_async_failing_tool() -> None:
    """Test ToolRetryMiddlewarewith async execution and failing tool."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "failing_tool" in tool_messages[0].content
    assert "3 attempts" in tool_messages[0].content
    assert tool_messages[0].status == "error"


async def test_tool_retry_async_succeeds_after_retries() -> None:
    """Test ToolRetryMiddlewareasync execution succeeds after temporary failures."""
    temp_fail = TemporaryFailureTool(fail_count=2)

    @tool
    def temp_failing_tool(value: str) -> str:
        """Tool that fails temporarily."""
        return temp_fail(value)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="temp_failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(
        max_retries=3,
        initial_delay=0.01,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[temp_failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use temp failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Success after 3 attempts" in tool_messages[0].content


async def test_tool_retry_async_backoff_timing() -> None:
    """Test ToolRetryMiddlewareasync applies correct backoff delays."""
    temp_fail = TemporaryFailureTool(fail_count=3)

    @tool
    def temp_failing_tool(value: str) -> str:
        """Tool that fails temporarily."""
        return temp_fail(value)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="temp_failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(
        max_retries=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[temp_failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    start_time = time.time()
    result = await agent.ainvoke(
        {"messages": [HumanMessage("Use temp failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )
    elapsed = time.time() - start_time

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1

    # Expected delays: 0.1 + 0.2 + 0.4 = 0.7 seconds
    assert elapsed >= 0.6, f"Expected at least 0.6s, got {elapsed}s"


def test_tool_retry_zero_retries() -> None:
    """Test ToolRetryMiddlewarewith max_retries=0 (no retries)."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(
        max_retries=0,  # No retries
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    # Should fail after 1 attempt (no retries)
    assert "1 attempt" in tool_messages[0].content
    assert tool_messages[0].status == "error"


def test_tool_retry_multiple_middleware_composition() -> None:
    """Test ToolRetryMiddlewarecomposes correctly with other middleware."""
    call_log = []

    # Custom middleware that logs calls
    @wrap_tool_call
    def logging_middleware(
        request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]
    ) -> ToolMessage | Command[Any]:
        if request.tool:
            call_log.append(f"before_{request.tool.name}")
        response = handler(request)
        if request.tool:
            call_log.append(f"after_{request.tool.name}")
        return response

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    retry = ToolRetryMiddleware(max_retries=2, initial_delay=0.01, jitter=False)

    agent = create_agent(
        model=model,
        tools=[working_tool],
        middleware=[logging_middleware, retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use working tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Both middleware should be called
    assert call_log == ["before_working_tool", "after_working_tool"]

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Success: test" in tool_messages[0].content


def test_tool_retry_deprecated_raise_keyword() -> None:
    """Test ToolRetryMiddleware with deprecated 'raise' keyword shows deprecation warning."""
    with pytest.warns(DeprecationWarning, match="on_failure='raise' is deprecated"):
        retry = ToolRetryMiddleware(
            max_retries=2,
            on_failure="raise",  # type: ignore[arg-type]
        )

    # Should be converted to 'error'
    assert retry.on_failure == "error"


def test_tool_retry_deprecated_return_message_keyword() -> None:
    """Test tool retry with deprecated 'return_message' keyword.

    Test ToolRetryMiddleware with deprecated 'return_message' keyword shows deprecation
    warning.
    """
    # Use string concatenation to avoid batch replace affecting test code
    deprecated_value = "return" + "_message"
    with pytest.warns(DeprecationWarning, match="on_failure='return_message' is deprecated"):
        retry = ToolRetryMiddleware(
            max_retries=2,
            on_failure=deprecated_value,  # type: ignore[arg-type]
        )

    # Should be converted to 'continue'
    assert retry.on_failure == "continue"


def test_tool_retry_deprecated_raise_behavior() -> None:
    """Test ToolRetryMiddleware with deprecated 'raise' forwards to 'error' behavior."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    with pytest.warns(DeprecationWarning, match="on_failure='raise' is deprecated"):
        retry = ToolRetryMiddleware(
            max_retries=2,
            initial_delay=0.01,
            jitter=False,
            on_failure="raise",  # type: ignore[arg-type]
        )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    # Should raise the ValueError from the tool (same as 'error')
    with pytest.raises(ValueError, match="Failed: test"):
        agent.invoke(
            {"messages": [HumanMessage("Use failing tool")]},
            {"configurable": {"thread_id": "test"}},
        )


def test_tool_retry_deprecated_return_message_behavior() -> None:
    """Test ToolRetryMiddleware with deprecated 'return_message' forwards to 'continue' behavior."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    # Use string concatenation to avoid batch replace affecting test code
    deprecated_value = "return" + "_message"
    with pytest.warns(DeprecationWarning, match="on_failure='return_message' is deprecated"):
        retry = ToolRetryMiddleware(
            max_retries=2,
            initial_delay=0.01,
            jitter=False,
            on_failure=deprecated_value,  # type: ignore[arg-type]
        )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Use failing tool")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    # Should contain error message (same as 'continue')
    assert "failing_tool" in tool_messages[0].content
    assert "3 attempts" in tool_messages[0].content
    assert "ValueError" in tool_messages[0].content
    assert tool_messages[0].status == "error"
