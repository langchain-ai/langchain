"""Unit tests for tool call interceptor in ToolNode."""

from collections.abc import Callable
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from langgraph.types import Command

from langchain.tools.tool_node import (
    ToolCallRequest,
    _ToolNode,
)

pytestmark = pytest.mark.anyio


def _create_mock_runtime(store: BaseStore | None = None) -> Mock:
    mock_runtime = Mock()
    mock_runtime.store = store
    mock_runtime.context = None
    mock_runtime.stream_writer = lambda _: None
    return mock_runtime


def _create_config_with_runtime(store: BaseStore | None = None) -> RunnableConfig:
    return {"configurable": {"__pregel_runtime": _create_mock_runtime(store)}}


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
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Simple passthrough handler."""
        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=passthrough_handler)

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
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "3"
    assert tool_message.tool_call_id == "call_1"
    assert tool_message.status != "error"


async def test_passthrough_handler_async() -> None:
    """Test passthrough handler with async tool."""

    def passthrough_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Simple passthrough handler."""
        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=passthrough_handler)

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
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "5"
    assert tool_message.tool_call_id == "call_2"


def test_modify_arguments() -> None:
    """Test handler that modifies tool arguments before execution."""

    def modify_args_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that doubles the input arguments."""
        # Modify the arguments
        request.tool_call["args"]["a"] *= 2
        request.tool_call["args"]["b"] *= 2

        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=modify_args_handler)

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
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    # Original args were (1, 2), doubled to (2, 4), so result is 6
    assert tool_message.content == "6"


def test_handler_validation_no_return() -> None:
    """Test that handler must return a result."""

    def handler_with_explicit_none(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that executes and returns result."""
        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=handler_with_explicit_none)

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
        },
        config=_create_config_with_runtime(),
    )

    assert isinstance(result, dict)
    messages = result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].content == "3"


def test_handler_validation_no_yield() -> None:
    """Test that handler that doesn't call execute returns None (bad behavior)."""

    def bad_handler(
        _request: ToolCallRequest,
        _execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that doesn't call execute - will cause type error."""
        # Don't call execute, just return None (invalid)
        return None  # type: ignore[return-value]

    tool_node = _ToolNode([add], wrap_tool_call=bad_handler)

    # This will return None wrapped in messages
    result = tool_node.invoke(
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
        },
        config=_create_config_with_runtime(),
    )

    # Result contains None in messages (bad handler behavior)
    assert isinstance(result, dict)
    assert result["messages"][0] is None


def test_handler_with_handle_tool_errors_true() -> None:
    """Test that handle_tool_errors=True works with on_tool_call handler."""

    def passthrough_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Simple passthrough handler."""
        message = execute(request)
        # When handle_tool_errors=True, errors should be converted to error messages
        assert isinstance(message, ToolMessage)
        assert message.status == "error"
        return message

    tool_node = _ToolNode(
        [failing_tool], wrap_tool_call=passthrough_handler, handle_tool_errors=True
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
                            "id": "call_9",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "error"


def test_multiple_tool_calls_with_handler() -> None:
    """Test handler with multiple tool calls in one message."""
    call_count = 0

    def counting_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that counts calls."""
        nonlocal call_count
        call_count += 1
        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=counting_handler)

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
        },
        config=_create_config_with_runtime(),
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
    state: dict = {"messages": []}
    runtime = None

    request = ToolCallRequest(tool_call=tool_call, tool=add, state=state, runtime=runtime)

    assert request.tool_call == tool_call
    assert request.tool == add
    assert request.state == state
    assert request.runtime is None
    assert request.tool_call["name"] == "add"


async def test_handler_with_async_execution() -> None:
    """Test handler works correctly with async tool execution."""

    @tool
    def async_add(a: int, b: int) -> int:
        """Async add two numbers."""
        return a + b

    def modifying_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that modifies arguments."""
        # Add 10 to both arguments
        request.tool_call["args"]["a"] += 10
        request.tool_call["args"]["b"] += 10
        return execute(request)

    tool_node = _ToolNode([async_add], wrap_tool_call=modifying_handler)

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
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    # Original: 1 + 2 = 3, with modifications: 11 + 12 = 23
    assert tool_message.content == "23"


def test_short_circuit_with_tool_message() -> None:
    """Test handler that returns ToolMessage to short-circuit tool execution."""

    def short_circuit_handler(
        request: ToolCallRequest,
        _execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that returns cached result without executing tool."""
        # Return a ToolMessage directly instead of calling execute
        return ToolMessage(
            content="cached_result",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    tool_node = _ToolNode([add], wrap_tool_call=short_circuit_handler)

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
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "cached_result"
    assert tool_message.tool_call_id == "call_16"
    assert tool_message.name == "add"


async def test_short_circuit_with_tool_message_async() -> None:
    """Test async handler that returns ToolMessage to short-circuit tool execution."""

    def short_circuit_handler(
        request: ToolCallRequest,
        _execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that returns cached result without executing tool."""
        return ToolMessage(
            content="async_cached_result",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    tool_node = _ToolNode([add], wrap_tool_call=short_circuit_handler)

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
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "async_cached_result"
    assert tool_message.tool_call_id == "call_17"


def test_conditional_short_circuit() -> None:
    """Test handler that conditionally short-circuits based on request."""
    call_count = {"count": 0}

    def conditional_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that caches even numbers, executes odd."""
        call_count["count"] += 1
        a = request.tool_call["args"]["a"]

        if a % 2 == 0:
            # Even: use cached result
            return ToolMessage(
                content=f"cached_{a}",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )
        # Odd: execute normally
        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=conditional_handler)

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
        },
        config=_create_config_with_runtime(),
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
        },
        config=_create_config_with_runtime(),
    )

    tool_message2 = result2["messages"][-1]
    assert tool_message2.content == "7"  # Actual execution: 3 + 4


def test_direct_return_tool_message() -> None:
    """Test handler that returns ToolMessage directly without calling execute."""

    def direct_return_handler(
        request: ToolCallRequest,
        _execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that returns ToolMessage directly."""
        # Return ToolMessage directly instead of calling execute
        return ToolMessage(
            content="direct_return",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    tool_node = _ToolNode([add], wrap_tool_call=direct_return_handler)

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
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "direct_return"
    assert tool_message.tool_call_id == "call_21"
    assert tool_message.name == "add"


async def test_direct_return_tool_message_async() -> None:
    """Test async handler that returns ToolMessage directly without calling execute."""

    def direct_return_handler(
        request: ToolCallRequest,
        _execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that returns ToolMessage directly."""
        return ToolMessage(
            content="async_direct_return",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    tool_node = _ToolNode([add], wrap_tool_call=direct_return_handler)

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
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "async_direct_return"
    assert tool_message.tool_call_id == "call_22"


def test_conditional_direct_return() -> None:
    """Test handler that conditionally returns ToolMessage directly or executes tool."""

    def conditional_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that returns cached or executes based on condition."""
        a = request.tool_call["args"]["a"]

        if a == 0:
            # Return ToolMessage directly for zero
            return ToolMessage(
                content="zero_cached",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )
        # Execute tool normally
        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=conditional_handler)

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
        },
        config=_create_config_with_runtime(),
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
        },
        config=_create_config_with_runtime(),
    )

    tool_message2 = result2["messages"][-1]
    assert tool_message2.content == "7"  # Actual execution: 3 + 4


def test_handler_can_throw_exception() -> None:
    """Test that a handler can throw an exception to signal error."""

    def throwing_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that throws an exception after receiving response."""
        response = execute(request)
        # Check response and throw if invalid
        if isinstance(response, ToolMessage):
            msg = "Handler rejected the response"
            raise TypeError(msg)
        return response

    tool_node = _ToolNode([add], wrap_tool_call=throwing_handler, handle_tool_errors=True)

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
        },
        config=_create_config_with_runtime(),
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
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that throws an exception."""
        execute(request)
        msg = "Handler error"
        raise ValueError(msg)

    tool_node = _ToolNode([add], wrap_tool_call=throwing_handler, handle_tool_errors=False)

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
            },
            config=_create_config_with_runtime(),
        )


def test_retry_middleware_with_exception() -> None:
    """Test retry middleware pattern that can call execute multiple times."""
    attempt_count = {"count": 0}

    def retry_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that can retry by calling execute multiple times."""
        max_retries = 3

        for _attempt in range(max_retries):
            attempt_count["count"] += 1
            response = execute(request)

            # Simulate checking for retriable errors
            # In real use case, would check response.status or content
            if isinstance(response, ToolMessage):
                # For this test, just succeed immediately
                return response

        # If we exhausted retries, return last response
        return response

    tool_node = _ToolNode([add], wrap_tool_call=retry_handler)

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
        },
        config=_create_config_with_runtime(),
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
        _request: ToolCallRequest,
        _execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that throws an exception before calling execute."""
        # Throw exception before executing (to avoid async/await complications)
        msg = "Async handler rejected the request"
        raise ValueError(msg)

    tool_node = _ToolNode([add], wrap_tool_call=throwing_handler, handle_tool_errors=True)

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
        },
        config=_create_config_with_runtime(),
    )

    # Should get error message due to handle_tool_errors=True
    messages = result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].status == "error"
    assert "Async handler rejected the request" in messages[0].content


def test_handler_cannot_yield_multiple_tool_messages() -> None:
    """Test that handler can only return once (not applicable to handler pattern)."""
    # With handler pattern, you can only return once by definition
    # This test is no longer relevant - handlers naturally return once
    # Keep test for compatibility but with simple passthrough

    def single_return_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that returns once (as all handlers do)."""
        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=single_return_handler)

    result = tool_node.invoke(
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
        },
        config=_create_config_with_runtime(),
    )

    # Should succeed - handlers can only return once
    assert isinstance(result, dict)
    assert len(result["messages"]) == 1


def test_handler_cannot_yield_request_after_tool_message() -> None:
    """Test that handler pattern doesn't allow multiple returns (not applicable)."""
    # With handler pattern, you can only return once
    # This test is no longer relevant

    def single_return_handler(
        request: ToolCallRequest,
        _execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that returns cached result."""
        # Return cached result (short-circuit)
        return ToolMessage("cached", tool_call_id=request.tool_call["id"], name="add")

    tool_node = _ToolNode([add], wrap_tool_call=single_return_handler)

    result = tool_node.invoke(
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
        },
        config=_create_config_with_runtime(),
    )

    # Should succeed with cached result
    assert isinstance(result, dict)
    assert result["messages"][0].content == "cached"


def test_handler_can_short_circuit_with_command() -> None:
    """Test that handler can short-circuit by returning Command."""

    def command_handler(
        _request: ToolCallRequest,
        _execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that short-circuits with Command."""
        # Short-circuit with Command instead of executing tool
        return Command(goto="end")

    tool_node = _ToolNode([add], wrap_tool_call=command_handler)

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
        },
        config=_create_config_with_runtime(),
    )

    # Should get Command in result list
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Command)
    assert result[0].goto == "end"


def test_handler_cannot_yield_multiple_commands() -> None:
    """Test that handler can only return once (not applicable to handler pattern)."""
    # With handler pattern, you can only return once
    # This test is no longer relevant

    def single_command_handler(
        _request: ToolCallRequest,
        _execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that returns Command once."""
        return Command(goto="step1")

    tool_node = _ToolNode([add], wrap_tool_call=single_command_handler)

    result = tool_node.invoke(
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
        },
        config=_create_config_with_runtime(),
    )

    # Should succeed - handlers naturally return once
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Command)
    assert result[0].goto == "step1"


def test_handler_cannot_yield_request_after_command() -> None:
    """Test that handler can only return once (not applicable to handler pattern)."""
    # With handler pattern, you can only return once
    # This test is no longer relevant

    def command_handler(
        _request: ToolCallRequest,
        _execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that returns Command."""
        return Command(goto="somewhere")

    tool_node = _ToolNode([add], wrap_tool_call=command_handler)

    result = tool_node.invoke(
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
        },
        config=_create_config_with_runtime(),
    )

    # Should succeed with Command
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Command)
    assert result[0].goto == "somewhere"


def test_tool_returning_command_sent_to_handler() -> None:
    """Test that when tool returns Command, it's sent to handler."""
    received_commands = []

    def command_inspector_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that inspects Command returned by tool."""
        result = execute(request)
        # Should receive Command from tool
        if isinstance(result, Command):
            received_commands.append(result)
        return result

    tool_node = _ToolNode([command_tool], wrap_tool_call=command_inspector_handler)

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
        },
        config=_create_config_with_runtime(),
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
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that modifies Command returned by tool."""
        result = execute(request)
        # Modify the Command
        if isinstance(result, Command):
            return Command(goto=f"modified_{result.goto}")
        return result

    tool_node = _ToolNode([command_tool], wrap_tool_call=command_modifier_handler)

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
        },
        config=_create_config_with_runtime(),
    )

    # Final result should be the modified Command in result list
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Command)
    assert result[0].goto == "modified_original"


def test_state_extraction_with_dict_input() -> None:
    """Test that state is correctly passed when input is a dict."""
    state_seen = []

    def state_inspector_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that records the state it receives."""
        state_seen.append(request.state)
        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=state_inspector_handler)

    input_state = {
        "messages": [
            AIMessage(
                "test",
                tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"}],
            )
        ],
        "other_field": "value",
    }

    tool_node.invoke(input_state, config=_create_config_with_runtime())

    # State should be the dict we passed in
    assert len(state_seen) == 1
    assert state_seen[0] == input_state
    assert isinstance(state_seen[0], dict)
    assert "messages" in state_seen[0]
    assert "other_field" in state_seen[0]
    assert "__type" not in state_seen[0]


def test_state_extraction_with_list_input() -> None:
    """Test that state is correctly passed when input is a list."""
    state_seen = []

    def state_inspector_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that records the state it receives."""
        state_seen.append(request.state)
        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=state_inspector_handler)

    input_state = [
        AIMessage(
            "test",
            tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"}],
        )
    ]

    tool_node.invoke(input_state, config=_create_config_with_runtime())

    # State should be the list we passed in
    assert len(state_seen) == 1
    assert state_seen[0] == input_state
    assert isinstance(state_seen[0], list)


def test_state_extraction_with_tool_call_with_context() -> None:
    """Test that state is correctly extracted from ToolCallWithContext.

    This tests the scenario where ToolNode is invoked via the Send API in
    create_agent, which wraps the tool call with additional context including
    the graph state.
    """
    state_seen = []

    def state_inspector_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that records the state it receives."""
        state_seen.append(request.state)
        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=state_inspector_handler)

    # Simulate ToolCallWithContext as used by create_agent with Send API
    actual_state = {
        "messages": [AIMessage("test")],
        "thread_model_call_count": 1,
        "run_model_call_count": 1,
        "custom_field": "custom_value",
    }

    tool_call_with_context = {
        "__type": "tool_call_with_context",
        "tool_call": {"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1", "type": "tool_call"},
        "state": actual_state,
    }

    tool_node.invoke(tool_call_with_context, config=_create_config_with_runtime())

    # State should be the extracted state from ToolCallWithContext, not the wrapper
    assert len(state_seen) == 1
    assert state_seen[0] == actual_state
    assert isinstance(state_seen[0], dict)
    assert "messages" in state_seen[0]
    assert "thread_model_call_count" in state_seen[0]
    assert "custom_field" in state_seen[0]
    # Most importantly, __type should NOT be in the extracted state
    assert "__type" not in state_seen[0]
    # And tool_call should not be in the state
    assert "tool_call" not in state_seen[0]


async def test_state_extraction_with_tool_call_with_context_async() -> None:
    """Test that state is correctly extracted from ToolCallWithContext in async mode."""
    state_seen = []

    def state_inspector_handler(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handler that records the state it receives."""
        state_seen.append(request.state)
        return execute(request)

    tool_node = _ToolNode([add], wrap_tool_call=state_inspector_handler)

    # Simulate ToolCallWithContext as used by create_agent with Send API
    actual_state = {
        "messages": [AIMessage("test")],
        "thread_model_call_count": 1,
        "run_model_call_count": 1,
    }

    tool_call_with_context = {
        "__type": "tool_call_with_context",
        "tool_call": {"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1", "type": "tool_call"},
        "state": actual_state,
    }

    await tool_node.ainvoke(tool_call_with_context, config=_create_config_with_runtime())

    # State should be the extracted state from ToolCallWithContext
    assert len(state_seen) == 1
    assert state_seen[0] == actual_state
    assert "__type" not in state_seen[0]
    assert "tool_call" not in state_seen[0]
