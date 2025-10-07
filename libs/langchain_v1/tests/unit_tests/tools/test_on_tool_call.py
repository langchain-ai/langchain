"""Unit tests for on_tool_call handler in ToolNode."""

from collections.abc import Generator
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import tool

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


def test_passthrough_handler() -> None:
    """Test a simple passthrough handler that doesn't modify anything."""

    def passthrough_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest, ToolMessage, None]:
        """Simple passthrough handler."""
        message = yield request

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
    ) -> Generator[ToolCallRequest, ToolMessage, None]:
        """Simple passthrough handler."""
        response = yield request

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
    ) -> Generator[ToolCallRequest, ToolMessage, None]:
        """Handler that doubles the input arguments."""
        # Modify the arguments
        request.tool_call["args"]["a"] *= 2
        request.tool_call["args"]["b"] *= 2

        response = yield request

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
    ) -> Generator[ToolCallRequest, ToolMessage, None]:
        """Handler that returns None explicitly - should still work."""
        yield request
        # Explicit None return - protocol uses last sent message as result
        return None  # type: ignore[return-value]

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
    ) -> Generator[ToolCallRequest, ToolMessage, None]:
        """Handler that ends immediately without yielding."""
        # End immediately without yielding anything
        # Need unreachable yield to make this a generator function
        if False:
            yield request  # type: ignore[unreachable]
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
    ) -> Generator[ToolCallRequest, ToolMessage, None]:
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
    ) -> Generator[ToolCallRequest, ToolMessage, None]:
        """Handler that counts calls."""
        nonlocal call_count
        call_count += 1
        response = yield request

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
    ) -> Generator[ToolCallRequest, ToolMessage, None]:
        """Handler that modifies arguments."""
        # Add 10 to both arguments
        request.tool_call["args"]["a"] += 10
        request.tool_call["args"]["b"] += 10
        response = yield request

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
    ) -> Generator[ToolCallRequest | ToolMessage, ToolMessage, None]:
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
    ) -> Generator[ToolCallRequest | ToolMessage, ToolMessage, None]:
        """Handler that returns cached result without executing tool."""
        cached_result = ToolMessage(
            content="async_cached_result",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )
        response = yield cached_result

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
    ) -> Generator[ToolCallRequest | ToolMessage, ToolMessage, None]:
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
            response = yield cached
        else:
            # Odd: execute normally
            response = yield request

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


def test_short_circuit_then_retry() -> None:
    """Test handler that yields ToolMessage then retries with actual tool."""
    attempt_count = {"count": 0}

    def cache_then_retry_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage, ToolMessage, None]:
        """Try cached result first, then execute tool if needed."""
        attempt_count["count"] += 1

        # First attempt: try cached result
        if attempt_count["count"] == 1:
            cached = ToolMessage(
                content="stale_cache",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )
            response = yield cached
            # Simulate cache validation failure, need fresh result
            # Yield the actual request to execute the tool
            response = yield request
        else:
            # Subsequent calls: execute normally
            response = yield request

    tool_node = ToolNode([add], on_tool_call=cache_then_retry_handler)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 5, "b": 6},
                            "id": "call_20",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    # Should have the actual result from the tool, not the cached value
    assert tool_message.content == "11"
    assert attempt_count["count"] == 1


def test_direct_return_tool_message() -> None:
    """Test handler that returns ToolMessage directly without yielding."""

    def direct_return_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage, ToolMessage, None]:
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
    ) -> Generator[ToolCallRequest | ToolMessage, ToolMessage, None]:
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
    ) -> Generator[ToolCallRequest, ToolMessage, None]:
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
            message = yield request

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
