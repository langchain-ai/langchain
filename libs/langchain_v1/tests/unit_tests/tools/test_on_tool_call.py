"""Unit tests for on_tool_call handler in ToolNode."""

from collections.abc import Generator
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import tool

from langchain.tools.tool_node import (
    ToolCallRequest,
    ToolCallResponse,
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
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Simple passthrough handler."""
        response = yield request
        return response

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
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Simple passthrough handler."""
        response = yield request
        return response

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
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Handler that doubles the input arguments."""
        # Modify the arguments
        request.tool_call["args"]["a"] *= 2
        request.tool_call["args"]["b"] *= 2

        response = yield request
        return response

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


def test_error_to_message_handler() -> None:
    """Test handler that converts errors to successful ToolMessage."""

    def error_to_message_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Handler that converts errors to messages."""
        try:
            response = yield request
            return response  # Success case
        except Exception as e:
            # Convert exception to error message
            error_message = ToolMessage(
                content=f"Error: {e}",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error",
            )
            return ToolCallResponse(
                action="return", result=error_message, exception=e
            )

    tool_node = ToolNode(
        [failing_tool], on_tool_call=error_to_message_handler, handle_tool_errors=False
    )

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "failing",
                    tool_calls=[
                        {
                            "name": "failing_tool",
                            "args": {"a": 1},
                            "id": "call_4",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert "Error:" in tool_message.content
    assert "This tool always fails" in tool_message.content
    assert tool_message.status == "error"


def test_retry_handler() -> None:
    """Test handler that retries with modified arguments on error."""
    attempt_count = 0

    def retry_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Handler that retries up to 2 times."""
        nonlocal attempt_count
        max_retries = 2

        for attempt in range(max_retries):
            attempt_count += 1
            try:
                response = yield request
                return response  # Success
            except Exception as e:
                # If this was the last retry, give up
                if attempt == max_retries - 1:
                    # Convert error to message
                    error_message = ToolMessage(
                        content=f"Failed after {max_retries} retries: {e}",
                        tool_call_id=request.tool_call["id"],
                        name=request.tool_call["name"],
                        status="error",
                    )
                    return ToolCallResponse(
                        action="return", result=error_message, exception=e
                    )

                # Otherwise, try with different args (won't help in this case, but demonstrates retry)
                request.tool_call["args"]["a"] += 1

        # This should never be reached
        msg = "Unreachable"
        raise RuntimeError(msg)

    tool_node = ToolNode([failing_tool], on_tool_call=retry_handler, handle_tool_errors=False)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "failing",
                    tool_calls=[
                        {
                            "name": "failing_tool",
                            "args": {"a": 1},
                            "id": "call_5",
                        }
                    ],
                )
            ]
        }
    )

    # Verify we attempted 2 times
    assert attempt_count == 2

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert "Failed after 2 retries" in tool_message.content
    assert tool_message.status == "error"


def test_handler_validation_no_return() -> None:
    """Test that handler without explicit return raises error."""

    def bad_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Handler that doesn't return explicitly."""
        yield request
        # Implicit None return
        return None  # type: ignore[return-value]

    tool_node = ToolNode([add], on_tool_call=bad_handler)

    with pytest.raises(ValueError, match="must explicitly return a ToolCallResponse"):
        tool_node.invoke(
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


def test_handler_validation_no_yield() -> None:
    """Test that handler must yield at least once."""

    def bad_handler(request: ToolCallRequest, _state: Any, _runtime: Any) -> ToolCallResponse:
        """Handler that doesn't yield - not even a generator."""
        return ToolCallResponse(
            action="return",
            result=ToolMessage(content="fake", tool_call_id=request.tool_call["id"]),
        )

    tool_node = ToolNode([add], on_tool_call=bad_handler)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError)):
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


def test_handler_can_raise_error() -> None:
    """Test handler that propagates errors by not catching exceptions."""

    def raise_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Handler that propagates errors by not catching them."""
        response = yield request  # Exception will propagate if there's an error
        return response  # Only reached if no exception

    tool_node = ToolNode([failing_tool], on_tool_call=raise_handler, handle_tool_errors=False)

    with pytest.raises(ValueError, match="This tool always fails"):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "failing",
                        tool_calls=[
                            {
                                "name": "failing_tool",
                                "args": {"a": 1},
                                "id": "call_8",
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
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Simple passthrough handler."""
        response = yield request
        # When handle_tool_errors=True, errors should be converted to error messages
        assert response.action == "return"
        assert response.result is not None
        assert isinstance(response.result, ToolMessage)
        assert response.result.status == "error"
        return response

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
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Handler that counts calls."""
        nonlocal call_count
        call_count += 1
        response = yield request
        return response

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


def test_tool_call_response_dataclass_validation() -> None:
    """Test ToolCallResponse validation."""
    # Valid continue response
    continue_response = ToolCallResponse(
        action="return",
        result=ToolMessage(content="success", tool_call_id="1"),
    )
    assert continue_response.action == "return"
    assert continue_response.result is not None

    # Valid raise response
    raise_response = ToolCallResponse(
        action="raise",
        exception=ValueError("error"),
    )
    assert raise_response.action == "raise"
    assert raise_response.exception is not None

    # Invalid: continue without result
    with pytest.raises(ValueError, match="action='return' requires a result"):
        ToolCallResponse(action="return")

    # Invalid: raise without exception
    with pytest.raises(ValueError, match="action='raise' requires an exception"):
        ToolCallResponse(action="raise")


async def test_handler_with_async_execution() -> None:
    """Test handler works correctly with async tool execution."""

    @tool
    async def async_add(a: int, b: int) -> int:
        """Async add two numbers."""
        return a + b

    def modifying_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Handler that modifies arguments."""
        # Add 10 to both arguments
        request.tool_call["args"]["a"] += 10
        request.tool_call["args"]["b"] += 10
        response = yield request
        return response

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


def test_response_validation_action_continue() -> None:
    """Test ToolCallResponse validation for action='return'."""

    def bad_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Handler that returns invalid response."""
        yield request
        # Return continue without result - this should be caught by validation
        return ToolCallResponse(action="return", result=None)

    tool_node = ToolNode([add], on_tool_call=bad_handler)

    with pytest.raises(ValueError, match="action='return' requires a result"):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "adding",
                        tool_calls=[
                            {
                                "name": "add",
                                "args": {"a": 1, "b": 2},
                                "id": "call_14",
                            }
                        ],
                    )
                ]
            }
        )


def test_response_validation_action_raise() -> None:
    """Test ToolCallResponse validation for action='raise'."""

    def bad_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
        """Handler that returns invalid response."""
        yield request
        # Return raise without exception
        return ToolCallResponse(action="raise", exception=None)

    tool_node = ToolNode([add], on_tool_call=bad_handler)

    with pytest.raises(ValueError, match="action='raise' requires an exception"):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "adding",
                        tool_calls=[
                            {
                                "name": "add",
                                "args": {"a": 1, "b": 2},
                                "id": "call_15",
                            }
                        ],
                    )
                ]
            }
        )


def test_short_circuit_with_tool_message() -> None:
    """Test handler that yields ToolMessage to short-circuit tool execution."""

    def short_circuit_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage, ToolCallResponse, ToolCallResponse]:
        """Handler that returns cached result without executing tool."""
        # Yield a ToolMessage directly instead of a ToolCallRequest
        cached_result = ToolMessage(
            content="cached_result",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )
        response = yield cached_result
        # Response should contain our cached message
        assert response.action == "return"
        assert response.result == cached_result
        return response

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
    ) -> Generator[ToolCallRequest | ToolMessage, ToolCallResponse, ToolCallResponse]:
        """Handler that returns cached result without executing tool."""
        cached_result = ToolMessage(
            content="async_cached_result",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )
        response = yield cached_result
        return response

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
    ) -> Generator[ToolCallRequest | ToolMessage, ToolCallResponse, ToolCallResponse]:
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
            return response
        else:
            # Odd: execute normally
            response = yield request
            return response

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
    ) -> Generator[ToolCallRequest | ToolMessage, ToolCallResponse, ToolCallResponse]:
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
            return response
        else:
            # Subsequent calls: execute normally
            response = yield request
            return response

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
    ) -> Generator[ToolCallRequest | ToolMessage, ToolCallResponse, ToolCallResponse | ToolMessage]:
        """Handler that returns ToolMessage directly."""
        # Return ToolMessage directly - should be auto-wrapped in ToolCallResponse
        # Note: We still need this to be a generator, so we use return (not yield)
        # The generator protocol will catch the StopIteration with the return value
        if False:
            yield  # Makes this a generator function
        return ToolMessage(
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
    ) -> Generator[ToolCallRequest | ToolMessage, ToolCallResponse, ToolCallResponse | ToolMessage]:
        """Handler that returns ToolMessage directly."""
        if False:
            yield  # Makes this a generator function
        return ToolMessage(
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
    ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse | ToolMessage]:
        """Handler that returns cached or executes based on condition."""
        a = request.tool_call["args"]["a"]

        if a == 0:
            # Return ToolMessage directly for zero
            return ToolMessage(
                content="zero_cached",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )
        else:
            # Execute tool normally
            response = yield request
            return response

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

def test_exception_thrown_into_generator() -> None:
    """Test that exceptions are thrown into generator when tool fails."""

    def exception_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage, ToolCallResponse, ToolCallResponse | ToolMessage]:
        """Handler that catches exceptions."""
        try:
            response = yield request
            return response
        except Exception as e:
            # Catch exception and return error message
            return ToolMessage(
                content=f"Caught: {e}",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error",
            )

    tool_node = ToolNode([failing_tool], on_tool_call=exception_handler, handle_tool_errors=False)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "failing",
                    tool_calls=[
                        {
                            "name": "failing_tool",
                            "args": {"a": 5},
                            "id": "call_25",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert "Caught:" in tool_message.content
    assert "This tool always fails" in tool_message.content
    assert tool_message.status == "error"


def test_exception_retry_with_throw() -> None:
    """Test retry logic using exception throwing."""
    attempt_count = {"count": 0}

    def retry_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage, ToolCallResponse, ToolCallResponse | ToolMessage]:
        """Handler that retries up to 3 times using exception handling."""
        for attempt in range(3):
            attempt_count["count"] += 1
            try:
                response = yield request
                return response  # Success
            except Exception as e:
                if attempt == 2:
                    # Give up, return error message
                    return ToolMessage(
                        content=f"Failed after 3 attempts: {e}",
                        tool_call_id=request.tool_call["id"],
                        name=request.tool_call["name"],
                        status="error",
                    )
                # Otherwise retry
        return ToolCallResponse(action="return", result=ToolMessage(content="unreachable", tool_call_id=""))

    tool_node = ToolNode([failing_tool], on_tool_call=retry_handler, handle_tool_errors=False)

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "failing",
                    tool_calls=[
                        {
                            "name": "failing_tool",
                            "args": {"a": 10},
                            "id": "call_26",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert "Failed after 3 attempts" in tool_message.content
    assert attempt_count["count"] == 3


def test_exception_propagation() -> None:
    """Test that uncaught exceptions propagate."""

    def no_catch_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage, ToolCallResponse, ToolCallResponse | ToolMessage]:
        """Handler that doesn't catch exceptions."""
        response = yield request
        return response

    tool_node = ToolNode([failing_tool], on_tool_call=no_catch_handler, handle_tool_errors=False)

    with pytest.raises(ValueError, match="This tool always fails"):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "failing",
                        tool_calls=[
                            {
                                "name": "failing_tool",
                                "args": {"a": 7},
                                "id": "call_27",
                            }
                        ],
                    )
                ]
            }
        )


async def test_exception_thrown_into_generator_async() -> None:
    """Test that exceptions are thrown into generator in async execution."""

    def exception_handler(
        request: ToolCallRequest, _state: Any, _runtime: Any
    ) -> Generator[ToolCallRequest | ToolMessage, ToolCallResponse, ToolCallResponse | ToolMessage]:
        """Handler that catches exceptions."""
        try:
            response = yield request
            return response
        except Exception as e:
            return ToolMessage(
                content=f"Async caught: {e}",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error",
            )

    tool_node = ToolNode([failing_tool], on_tool_call=exception_handler, handle_tool_errors=False)

    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "failing",
                    tool_calls=[
                        {
                            "name": "failing_tool",
                            "args": {"a": 8},
                            "id": "call_28",
                        }
                    ],
                )
            ]
        }
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert "Async caught:" in tool_message.content
    assert "This tool always fails" in tool_message.content
