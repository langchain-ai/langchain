"""Unit tests for on_tool_call handler in ToolNode."""

from collections.abc import Generator
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from langchain.tools.tool_node import (
    ToolCallRequest,
    ToolNode,
)

pytestmark = pytest.mark.anyio


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def failing_tool(a: int) -> int:
    """A tool that always fails."""
    msg = f"This tool always fails (input: {a})"
    raise ValueError(msg)


@tool
def command_tool(goto: str) -> Command:
    """A tool that returns a Command."""
    return Command(goto=goto)


def test_passthrough_handler() -> None:
    """Test a simple passthrough handler that doesn't modify anything."""

    def passthrough_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Simple passthrough handler."""
        yield request

    tool_node = ToolNode([add], on_tool_call=passthrough_handler)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_1",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "3"
    assert tool_message.tool_call_id == "call_1"
    assert tool_message.status != "error"


async def test_passthrough_handler_async() -> None:
    """Test passthrough handler with async tool."""

    def passthrough_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Simple passthrough handler."""
        yield request

    tool_node = ToolNode([add], on_tool_call=passthrough_handler)

    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 2, "b": 3},
                            "id": "call_2",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "5"
    assert tool_message.tool_call_id == "call_2"


def test_modify_arguments() -> None:
    """Test handler that modifies tool arguments before execution."""

    def modify_args_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that doubles the input arguments."""
        # Modify the arguments
        request.tool_call["args"]["a"] *= 2
        request.tool_call["args"]["b"] *= 2

        yield request

    tool_node = ToolNode([add], on_tool_call=modify_args_handler)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_3",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    # Original args were (1, 2), doubled to (2, 4), so result is 6
    assert tool_message.content == "6"


def test_handler_validation_no_return() -> None:
    """Test that handler with explicit None return works (returns last sent message)."""

    def handler_with_explicit_none(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that returns None explicitly - should still work."""
        yield request
        # Explicit None return - protocol uses last sent message as result
        return None

    tool_node = ToolNode([add], on_tool_call=handler_with_explicit_none)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_6",
                        }
                    ],
                )
            ]
        }
    )

    assert isinstance(result, dict)
    messages = result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].content == "3"


def test_handler_validation_no_yield() -> None:
    """Test that handler must yield at least once."""

    def bad_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that ends immediately without yielding."""
        # End immediately without yielding anything
        # Need unreachable yield to make this a generator function
        if False:
            yield request
        return

    tool_node = ToolNode([add], on_tool_call=bad_handler)

    with pytest.raises(ValueError, match="must yield at least once"):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "adding",
                        tool_calls=[
                            {
                                "name": "add",
                                "args": {"a": 1, "b": 2},
                                "id": "call_7",
                            }
                        ],
                    )
                ]
            }
        )


def test_handler_with_handle_tool_errors_true() -> None:
    """Test that handle_tool_errors=True works with on_tool_call handler."""

    def passthrough_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Simple passthrough handler."""
        message = yield request
        # When handle_tool_errors=True, errors should be converted to error messages
        assert isinstance(message, ToolMessage)
        assert message.status == "error"

    tool_node = ToolNode([failing_tool], on_tool_call=passthrough_handler, handle_tool_errors=True)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "failing",
                    tool_calls=[
                        {
                            "name": "failing_tool",
                            "args": {"a": 1},
                            "id": "call_9",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "error"


def test_multiple_tool_calls_with_handler() -> None:
    """Test handler with multiple tool calls in one message."""
    call_count = 0

    def counting_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that counts calls."""
        nonlocal call_count
        call_count += 1
        yield request

    tool_node = ToolNode([add], on_tool_call=counting_handler)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding multiple",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_10",
                        },
                        {
                            "name": "add",
                            "args": {"a": 3, "b": 4},
                            "id": "call_11",
                        },
                        {
                            "name": "add",
                            "args": {"a": 5, "b": 6},
                            "id": "call_12",
                        },
                    ],
                )
            ]
        }
    )

    # Handler should be called once for each tool call
    assert call_count == 3

    # Verify all results
    messages = result["messages"]
    assert len(messages) == 3
    assert all(isinstance(m, ToolMessage) for m in messages)
    assert messages[0].content == "3"
    assert messages[1].content == "7"
    assert messages[2].content == "11"


def test_tool_call_request_dataclass() -> None:
    """Test ToolCallRequest dataclass."""
    tool_call: ToolCall = {"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"}

    request = ToolCallRequest(tool_call=tool_call, tool=add)

    assert request.tool_call == tool_call
    assert request.tool == add
    assert request.tool_call["name"] == "add"


async def test_handler_with_async_execution() -> None:
    """Test handler works correctly with async tool execution."""

    @tool
    async def async_add(a: int, b: int) -> int:
        """Async add two numbers."""
        return a + b

    def modifying_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that modifies arguments."""
        # Add 10 to both arguments
        request.tool_call["args"]["a"] += 10
        request.tool_call["args"]["b"] += 10
        yield request

    tool_node = ToolNode([async_add], on_tool_call=modifying_handler)

    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "async_add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_13",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    # Original: 1 + 2 = 3, with modifications: 11 + 12 = 23
    assert tool_message.content == "23"


def test_short_circuit_with_tool_message() -> None:
    """Test handler that yields ToolMessage to short-circuit tool execution."""

    def short_circuit_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that returns cached result without executing tool."""
        # Yield a ToolMessage directly instead of a ToolCallRequest
        cached_result = ToolMessage(
            content="cached_result",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )
        message = yield cached_result
        # Message should be our cached message sent back
        assert message == cached_result

    tool_node = ToolNode([add], on_tool_call=short_circuit_handler)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_16",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "cached_result"
    assert tool_message.tool_call_id == "call_16"
    assert tool_message.name == "add"


async def test_short_circuit_with_tool_message_async() -> None:
    """Test async handler that yields ToolMessage to short-circuit tool execution."""

    def short_circuit_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that returns cached result without executing tool."""
        cached_result = ToolMessage(
            content="async_cached_result",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )
        yield cached_result

    tool_node = ToolNode([add], on_tool_call=short_circuit_handler)

    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 2, "b": 3},
                            "id": "call_17",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "async_cached_result"
    assert tool_message.tool_call_id == "call_17"


def test_conditional_short_circuit() -> None:
    """Test handler that conditionally short-circuits based on request."""
    call_count = {"count": 0}

    def conditional_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that caches even numbers, executes odd."""
        call_count["count"] += 1
        a = request.tool_call["args"]["a"]

        if a % 2 == 0:
            # Even: use cached result
            cached = ToolMessage(
                content=f"cached_{a}",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )
            yield cached
        else:
            # Odd: execute normally
            yield request

    tool_node = ToolNode([add], on_tool_call=conditional_handler)

    # Test with even number (should be cached)
    result1 = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 2, "b": 3},
                            "id": "call_18",
                        }
                    ],
                )
            ]
        }
    )

    tool_message1 = result1["messages"][-1]
    assert tool_message1.content == "cached_2"

    # Test with odd number (should execute)
    result2 = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 3, "b": 4},
                            "id": "call_19",
                        }
                    ],
                )
            ]
        }
    )

    tool_message2 = result2["messages"][-1]
    assert tool_message2.content == "7"  # Actual execution: 3 + 4


def test_direct_return_tool_message() -> None:
    """Test handler that returns ToolMessage directly without yielding."""

    def direct_return_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that returns ToolMessage directly."""
        # Return ToolMessage directly
        # Note: We still need this to be a generator, so we use return (not yield)
        # The generator protocol will catch the StopIteration with the return value
        if False:
            yield  # Makes this a generator function
        yield ToolMessage(
            content="direct_return",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    tool_node = ToolNode([add], on_tool_call=direct_return_handler)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_21",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "direct_return"
    assert tool_message.tool_call_id == "call_21"
    assert tool_message.name == "add"


async def test_direct_return_tool_message_async() -> None:
    """Test async handler that returns ToolMessage directly without yielding."""

    def direct_return_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that returns ToolMessage directly."""
        if False:
            yield  # Makes this a generator function
        yield ToolMessage(
            content="async_direct_return",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    tool_node = ToolNode([add], on_tool_call=direct_return_handler)

    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 2, "b": 3},
                            "id": "call_22",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "async_direct_return"
    assert tool_message.tool_call_id == "call_22"


def test_conditional_direct_return() -> None:
    """Test handler that conditionally returns ToolMessage directly or executes tool."""

    def conditional_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that returns cached or executes based on condition."""
        a = request.tool_call["args"]["a"]

        if a == 0:
            # Return ToolMessage directly for zero
            if False:
                yield  # Makes this a generator
            yield ToolMessage(
                content="zero_cached",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )
        else:
            # Execute tool normally
            yield request

    tool_node = ToolNode([add], on_tool_call=conditional_handler)

    # Test with zero (should return directly)
    result1 = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 0, "b": 5},
                            "id": "call_23",
                        }
                    ],
                )
            ]
        }
    )

    tool_message1 = result1["messages"][-1]
    assert tool_message1.content == "zero_cached"

    # Test with non-zero (should execute)
    result2 = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 3, "b": 4},
                            "id": "call_24",
                        }
                    ],
                )
            ]
        }
    )

    tool_message2 = result2["messages"][-1]
    assert tool_message2.content == "7"  # Actual execution: 3 + 4


def test_handler_can_throw_exception() -> None:
    """Test that a handler can throw an exception to signal error."""

    def throwing_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that throws an exception after receiving response."""
        response = yield request
        # Check response and throw if invalid
        if isinstance(response, ToolMessage):
            msg = "Handler rejected the response"
            raise ValueError(msg)  # noqa: TRY004

    tool_node = ToolNode([add], on_tool_call=throwing_handler, handle_tool_errors=True)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_exc_1",
                        }
                    ],
                )
            ]
        }
    )

    # Should get error message due to handle_tool_errors=True
    messages = result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].status == "error"
    assert "Handler rejected the response" in messages[0].content


def test_handler_throw_without_handle_errors() -> None:
    """Test that exception propagates when handle_tool_errors=False."""

    def throwing_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that throws an exception."""
        yield request
        msg = "Handler error"
        raise ValueError(msg)

    tool_node = ToolNode([add], on_tool_call=throwing_handler, handle_tool_errors=False)

    with pytest.raises(ValueError, match="Handler error"):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "adding",
                        tool_calls=[
                            {
                                "name": "add",
                                "args": {"a": 1, "b": 2},
                                "id": "call_exc_2",
                            }
                        ],
                    )
                ]
            }
        )


def test_retry_middleware_with_exception() -> None:
    """Test retry middleware pattern that throws after exhausting retries."""
    attempt_count = {"count": 0}

    def retry_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that retries up to 3 times, then throws."""
        max_retries = 3

        for attempt in range(max_retries):
            attempt_count["count"] += 1
            response = yield request

            # Simulate checking for retriable errors
            # In real use case, would check response.status or content
            if isinstance(response, ToolMessage) and attempt < max_retries - 1:
                # Could retry based on some condition
                # For this test, just succeed immediately
                break

        # If we exhausted retries, could throw
        # For this test, we succeed on first try

    tool_node = ToolNode([add], on_tool_call=retry_handler)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_exc_3",
                        }
                    ],
                )
            ]
        }
    )

    # Should succeed after 1 attempt
    assert attempt_count["count"] == 1
    messages = result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].content == "3"


async def test_async_handler_can_throw_exception() -> None:
    """Test that async execution also supports exception throwing."""

    def throwing_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that throws an exception after receiving response."""
        response = yield request
        if isinstance(response, ToolMessage):
            msg = "Async handler rejected the response"
            raise ValueError(msg)  # noqa: TRY004

    tool_node = ToolNode([add], on_tool_call=throwing_handler, handle_tool_errors=True)

    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_exc_4",
                        }
                    ],
                )
            ]
        }
    )

    # Should get error message due to handle_tool_errors=True
    messages = result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].status == "error"
    assert "Async handler rejected the response" in messages[0].content


def test_handler_cannot_yield_multiple_tool_messages() -> None:
    """Test that yielding multiple ToolMessages is rejected."""

    def multi_message_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that incorrectly yields multiple ToolMessages."""
        # First short-circuit
        yield ToolMessage("first", tool_call_id=request.tool_call["id"], name="add")
        # Second short-circuit - should fail
        yield ToolMessage("second", tool_call_id=request.tool_call["id"], name="add")

    tool_node = ToolNode([add], on_tool_call=multi_message_handler)

    with pytest.raises(
        ValueError,
        match="on_tool_call handler yielded multiple values after short-circuit",
    ):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "adding",
                        tool_calls=[
                            {
                                "name": "add",
                                "args": {"a": 1, "b": 2},
                                "id": "call_multi_1",
                            }
                        ],
                    )
                ]
            }
        )


def test_handler_cannot_yield_request_after_tool_message() -> None:
    """Test that yielding ToolCallRequest after ToolMessage is rejected."""

    def confused_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that incorrectly switches from short-circuit to execution."""
        # First short-circuit with cached result
        yield ToolMessage("cached", tool_call_id=request.tool_call["id"], name="add")
        # Then try to execute - should fail
        yield request

    tool_node = ToolNode([add], on_tool_call=confused_handler)

    with pytest.raises(
        ValueError,
        match="on_tool_call handler yielded ToolCallRequest after short-circuit",
    ):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "adding",
                        tool_calls=[
                            {
                                "name": "add",
                                "args": {"a": 1, "b": 2},
                                "id": "call_confused_1",
                            }
                        ],
                    )
                ]
            }
        )


def test_handler_can_short_circuit_with_command() -> None:
    """Test that handler can short-circuit by yielding Command."""

    def command_handler(
        _request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that short-circuits with Command."""
        # Short-circuit with Command instead of executing tool
        yield Command(goto="end")

    tool_node = ToolNode([add], on_tool_call=command_handler)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_cmd_1",
                        }
                    ],
                )
            ]
        }
    )

    # Should get Command in result list
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Command)
    assert result[0].goto == "end"


def test_handler_cannot_yield_multiple_commands() -> None:
    """Test that yielding multiple Commands is rejected."""

    def multi_command_handler(
        _request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that incorrectly yields multiple Commands."""
        # First short-circuit
        yield Command(goto="step1")
        # Second short-circuit - should fail
        yield Command(goto="step2")

    tool_node = ToolNode([add], on_tool_call=multi_command_handler)

    with pytest.raises(
        ValueError,
        match="on_tool_call handler yielded multiple values after short-circuit",
    ):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "adding",
                        tool_calls=[
                            {
                                "name": "add",
                                "args": {"a": 1, "b": 2},
                                "id": "call_multicmd_1",
                            }
                        ],
                    )
                ]
            }
        )


def test_handler_cannot_yield_request_after_command() -> None:
    """Test that yielding ToolCallRequest after Command is rejected."""

    def command_then_request_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that incorrectly yields request after Command."""
        # First short-circuit with Command
        yield Command(goto="somewhere")
        # Then try to execute - should fail
        yield request

    tool_node = ToolNode([add], on_tool_call=command_then_request_handler)

    with pytest.raises(
        ValueError,
        match="on_tool_call handler yielded ToolCallRequest after short-circuit",
    ):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "adding",
                        tool_calls=[
                            {
                                "name": "add",
                                "args": {"a": 1, "b": 2},
                                "id": "call_cmdreq_1",
                            }
                        ],
                    )
                ]
            }
        )


def test_tool_returning_command_sent_to_handler() -> None:
    """Test that when tool returns Command, it's sent to handler."""
    received_commands = []

    def command_inspector_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that inspects Command returned by tool."""
        result = yield request
        # Should receive Command from tool
        if isinstance(result, Command):
            received_commands.append(result)
        # Can end here, returning the Command

    tool_node = ToolNode([command_tool], on_tool_call=command_inspector_handler)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "navigating",
                    tool_calls=[
                        {
                            "name": "command_tool",
                            "args": {"goto": "next_step"},
                            "id": "call_cmdtool_1",
                        }
                    ],
                )
            ]
        }
    )

    # Handler should have received the Command
    assert len(received_commands) == 1
    assert received_commands[0].goto == "next_step"

    # Final result should be the Command in result list
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Command)
    assert result[0].goto == "next_step"


def test_handler_can_modify_command_from_tool() -> None:
    """Test that handler can inspect and modify Command from tool."""

    def command_modifier_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Handler that modifies Command returned by tool."""
        result = yield request
        # Modify the Command
        if isinstance(result, Command):
            modified_cmd = Command(goto=f"modified_{result.goto}")
            yield modified_cmd
        # Otherwise pass through

    tool_node = ToolNode([command_tool], on_tool_call=command_modifier_handler)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "navigating",
                    tool_calls=[
                        {
                            "name": "command_tool",
                            "args": {"goto": "original"},
                            "id": "call_cmdmod_1",
                        }
                    ],
                )
            ]
        }
    )

    # Final result should be the modified Command in result list
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Command)
    assert result[0].goto == "modified_original"
