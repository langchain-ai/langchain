"""Tests for on_tool_call handler functionality."""

from collections.abc import Generator

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from langchain.tools import ToolNode
from langchain.tools.tool_node import ToolRequest, ToolResponse


# Test tools
@tool
def success_tool(x: int) -> int:
    """A tool that always succeeds."""
    return x * 2


@tool
def error_tool(x: int) -> int:
    """A tool that always raises ValueError."""
    msg = f"Error with value: {x}"
    raise ValueError(msg)


@tool
def rate_limit_tool(x: int) -> int:
    """A tool that simulates rate limit errors."""
    if not hasattr(rate_limit_tool, "_call_count"):
        rate_limit_tool._call_count = 0
    rate_limit_tool._call_count += 1

    if rate_limit_tool._call_count < 3:  # Fail first 2 times
        msg = "Rate limit exceeded"
        raise ValueError(msg)
    return x * 2


def test_on_tool_call_passthrough() -> None:
    """Test that a simple passthrough handler works."""

    def passthrough_handler(
        request: ToolRequest,
    ) -> Generator[ToolRequest, ToolResponse, ToolResponse]:
        """Simply pass through without modification."""
        response = yield request
        return response

    tool_node = ToolNode([success_tool], on_tool_call=passthrough_handler)
    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[{"name": "success_tool", "args": {"x": 5}, "id": "1"}],
                )
            ]
        }
    )

    assert len(result["messages"]) == 1
    tool_message: ToolMessage = result["messages"][0]
    assert tool_message.content == "10"
    assert tool_message.status != "error"


def test_on_tool_call_retry_success():
    """Test that retry handler can recover from transient errors."""
    # Reset counter
    if hasattr(rate_limit_tool, "_call_count"):
        rate_limit_tool._call_count = 0

    def retry_handler(
        request: ToolRequest,
    ) -> Generator[ToolRequest, ToolResponse, ToolResponse]:
        """Retry up to 3 times."""
        max_retries = 3

        for attempt in range(max_retries):
            response = yield request

            if response.action == "return":
                return response

            # Retry on error
            if attempt < max_retries - 1:
                continue

            # Final attempt failed - convert to error message
            return ToolResponse(
                action="return",
                result=ToolMessage(
                    content=f"Failed after {max_retries} attempts",
                    name=request.tool_call["name"],
                    tool_call_id=request.tool_call["id"],
                    status="error",
                ),
            )

        return response  # Should never reach here

    tool_node = ToolNode([rate_limit_tool], on_tool_call=retry_handler, handle_tool_errors=False)
    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[{"name": "rate_limit_tool", "args": {"x": 5}, "id": "1"}],
                )
            ]
        }
    )

    assert len(result["messages"]) == 1
    tool_message: ToolMessage = result["messages"][0]
    assert tool_message.content == "10"  # Should succeed on 3rd attempt
    assert tool_message.status != "error"


def test_on_tool_call_convert_error_to_message():
    """Test that handler can convert raised errors to error messages."""

    def error_to_message_handler(
        request: ToolRequest,
    ) -> Generator[ToolRequest, ToolResponse, ToolResponse]:
        """Convert any error to a user-friendly message."""
        response = yield request

        if response.action == "raise":
            return ToolResponse(
                action="return",
                result=ToolMessage(
                    content=f"Tool failed: {response.exception}",
                    name=request.tool_call["name"],
                    tool_call_id=request.tool_call["id"],
                    status="error",
                ),
                exception=response.exception,
            )

        return response

    tool_node = ToolNode(
        [error_tool], on_tool_call=error_to_message_handler, handle_tool_errors=False
    )
    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[{"name": "error_tool", "args": {"x": 5}, "id": "1"}],
                )
            ]
        }
    )

    assert len(result["messages"]) == 1
    tool_message: ToolMessage = result["messages"][0]
    assert "Tool failed" in tool_message.content
    assert "Error with value: 5" in tool_message.content
    assert tool_message.status == "error"


def test_on_tool_call_let_error_raise():
    """Test that handler can let errors propagate."""

    def let_raise_handler(
        request: ToolRequest,
    ) -> Generator[ToolRequest, ToolResponse, ToolResponse]:
        """Just return the response as-is, letting errors raise."""
        response = yield request
        return response

    tool_node = ToolNode([error_tool], on_tool_call=let_raise_handler, handle_tool_errors=False)

    with pytest.raises(ValueError) as exc_info:
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "",
                        tool_calls=[{"name": "error_tool", "args": {"x": 5}, "id": "1"}],
                    )
                ]
            }
        )

    assert "Error with value: 5" in str(exc_info.value)


def test_on_tool_call_with_handled_errors():
    """Test interaction between on_tool_call and handle_tool_errors."""
    call_count = {"count": 0}

    def counting_handler(
        request: ToolRequest,
    ) -> Generator[ToolRequest, ToolResponse, ToolResponse]:
        """Count how many times we're called."""
        call_count["count"] += 1
        response = yield request
        return response

    # When handle_tool_errors=True, errors are converted to ToolMessages
    # so handler sees action="return"
    tool_node = ToolNode([error_tool], on_tool_call=counting_handler, handle_tool_errors=True)
    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[{"name": "error_tool", "args": {"x": 5}, "id": "1"}],
                )
            ]
        }
    )

    assert call_count["count"] == 1
    assert len(result["messages"]) == 1
    tool_message: ToolMessage = result["messages"][0]
    assert tool_message.status == "error"
    assert "Please fix your mistakes" in tool_message.content


def test_on_tool_call_must_return_value():
    """Test that handler must return a ToolResponse."""

    def no_return_handler(
        request: ToolRequest,
    ) -> Generator[ToolRequest, ToolResponse, ToolResponse]:
        """Handler that doesn't return anything."""
        response = yield request
        # Implicit return None

    tool_node = ToolNode([success_tool], on_tool_call=no_return_handler)

    with pytest.raises(ValueError) as exc_info:
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "",
                        tool_calls=[{"name": "success_tool", "args": {"x": 5}, "id": "1"}],
                    )
                ]
            }
        )

    assert "must explicitly return a ToolResponse" in str(exc_info.value)


def test_on_tool_call_request_modification():
    """Test that handler can modify the request before execution."""

    def double_input_handler(
        request: ToolRequest,
    ) -> Generator[ToolRequest, ToolResponse, ToolResponse]:
        """Double the input value."""
        # Modify the tool call args
        modified_tool_call = {
            **request.tool_call,
            "args": {**request.tool_call["args"], "x": request.tool_call["args"]["x"] * 2},
        }
        modified_request = ToolRequest(
            tool_call=modified_tool_call,
            tool=request.tool,
            config=request.config,
        )
        response = yield modified_request
        return response

    tool_node = ToolNode([success_tool], on_tool_call=double_input_handler)
    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[{"name": "success_tool", "args": {"x": 5}, "id": "1"}],
                )
            ]
        }
    )

    assert len(result["messages"]) == 1
    tool_message: ToolMessage = result["messages"][0]
    # Input was 5, doubled to 10, then tool multiplies by 2 = 20
    assert tool_message.content == "20"


def test_on_tool_call_response_validation():
    """Test that ToolResponse validates action and required fields."""
    # Test action="return" requires result
    with pytest.raises(ValueError) as exc_info:
        ToolResponse(action="return")
    assert "action='return' requires a result" in str(exc_info.value)

    # Test action="raise" requires exception
    with pytest.raises(ValueError) as exc_info:
        ToolResponse(action="raise")
    assert "action='raise' requires an exception" in str(exc_info.value)

    # Valid responses should work
    ToolResponse(
        action="return",
        result=ToolMessage(content="test", tool_call_id="1", name="test"),
    )
    ToolResponse(action="raise", exception=ValueError("test"))


def test_on_tool_call_without_handler_backward_compat():
    """Test that tools work without on_tool_call handler (backward compatibility)."""
    # Success case
    tool_node = ToolNode([success_tool])
    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[{"name": "success_tool", "args": {"x": 5}, "id": "1"}],
                )
            ]
        }
    )
    assert result["messages"][0].content == "10"

    # Error case with handle_tool_errors=False
    tool_node_error = ToolNode([error_tool], handle_tool_errors=False)
    with pytest.raises(ValueError):
        tool_node_error.invoke(
            {
                "messages": [
                    AIMessage(
                        "",
                        tool_calls=[{"name": "error_tool", "args": {"x": 5}, "id": "1"}],
                    )
                ]
            }
        )

    # Error case with handle_tool_errors=True
    tool_node_handled = ToolNode([error_tool], handle_tool_errors=True)
    result = tool_node_handled.invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[{"name": "error_tool", "args": {"x": 5}, "id": "1"}],
                )
            ]
        }
    )
    assert result["messages"][0].status == "error"


def test_on_tool_call_multiple_yields():
    """Test that handler can yield multiple times for retries."""
    attempts = {"count": 0}

    def multi_yield_handler(
        request: ToolRequest,
    ) -> Generator[ToolRequest, ToolResponse, ToolResponse]:
        """Yield multiple times to track attempts."""
        max_attempts = 3

        for _ in range(max_attempts):
            attempts["count"] += 1
            response = yield request

            if response.action == "return":
                return response

        # All attempts failed
        return response

    tool_node = ToolNode([error_tool], on_tool_call=multi_yield_handler, handle_tool_errors=False)

    with pytest.raises(ValueError):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "",
                        tool_calls=[{"name": "error_tool", "args": {"x": 5}, "id": "1"}],
                    )
                ]
            }
        )

    assert attempts["count"] == 3
