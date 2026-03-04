import asyncio
import time
from typing import cast

import pytest
from langchain_core.messages import ToolMessage

from langchain.agents.middleware.tool_timeout import ToolTimeoutMiddleware
from langchain.agents.middleware.types import ToolCallRequest


class DummyRuntime:
    """Dummy runtime for testing."""


class DummyState:
    """Dummy state for testing."""


@pytest.fixture
def dummy_request() -> ToolCallRequest:
    """Return a valid ToolCallRequest mock."""
    return ToolCallRequest(
        tool_call={"name": "test_tool", "args": {"input": "test"}, "id": "call_123"},
        tool=None,  # type: ignore[arg-type]
        state={},  # type: ignore[arg-type]
        runtime=DummyRuntime(),  # type: ignore[arg-type]
    )


def test_initialization() -> None:
    """Test standard initialization and ValueError cases."""
    middleware = ToolTimeoutMiddleware(timeout=2.0)
    assert middleware.timeout == 2.0

    with pytest.raises(ValueError, match="safely greater than 0"):
        ToolTimeoutMiddleware(timeout=0.0)

    with pytest.raises(ValueError, match="safely greater than 0"):
        ToolTimeoutMiddleware(timeout=-1.5)


def test_sync_wrap_tool_call_success(dummy_request: ToolCallRequest) -> None:
    """Test successful synchronous execution within the timeout."""
    middleware = ToolTimeoutMiddleware(timeout=1.0)

    def fast_handler(req: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content="Success", name=req.tool_call["name"], tool_call_id=req.tool_call["id"]
        )

    result = middleware.wrap_tool_call(dummy_request, fast_handler)
    assert isinstance(result, ToolMessage)
    assert result.content == "Success"
    assert result.status != "error"


def test_sync_wrap_tool_call_timeout(dummy_request: ToolCallRequest) -> None:
    """Test synchronous execution that times out."""
    middleware = ToolTimeoutMiddleware(timeout=0.1)

    def slow_handler(req: ToolCallRequest) -> ToolMessage:
        time.sleep(0.5)
        return ToolMessage(
            content="Too Late", name=req.tool_call["name"], tool_call_id=req.tool_call["id"]
        )

    result = middleware.wrap_tool_call(dummy_request, slow_handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "timed out after 0.1 seconds" in cast("str", result.content)
    assert result.name == "test_tool"
    assert result.tool_call_id == "call_123"


@pytest.mark.asyncio
async def test_async_wrap_tool_call_success(dummy_request: ToolCallRequest) -> None:
    """Test successful asynchronous execution within the timeout."""
    middleware = ToolTimeoutMiddleware(timeout=1.0)

    async def async_fast_handler(req: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(0.01)
        return ToolMessage(
            content="Async Success", name=req.tool_call["name"], tool_call_id=req.tool_call["id"]
        )

    result = await middleware.awrap_tool_call(dummy_request, async_fast_handler)
    assert isinstance(result, ToolMessage)
    assert result.content == "Async Success"
    assert result.status != "error"


@pytest.mark.asyncio
async def test_async_wrap_tool_call_timeout(dummy_request: ToolCallRequest) -> None:
    """Test asynchronous execution that times out."""
    middleware = ToolTimeoutMiddleware(timeout=0.1)

    async def async_slow_handler(req: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(0.5)
        return ToolMessage(
            content="Too Late Async", name=req.tool_call["name"], tool_call_id=req.tool_call["id"]
        )

    result = await middleware.awrap_tool_call(dummy_request, async_slow_handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "timed out after 0.1 seconds" in cast("str", result.content)
    assert result.name == "test_tool"
    assert result.tool_call_id == "call_123"


def test_generate_timeout_error_handles_missing_keys() -> None:
    """Test _generate_timeout_error safely handles malformed tool_call dicts."""
    middleware = ToolTimeoutMiddleware(timeout=1.0)

    # Missing optional keys like id/name
    bad_request = ToolCallRequest(
        tool_call={},
        tool=None,  # type: ignore[arg-type]
        state={},  # type: ignore[arg-type]
        runtime=DummyRuntime(),  # type: ignore[arg-type]
    )

    msg = middleware._generate_timeout_error(bad_request)
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    assert msg.name == "unknown_tool"
    assert msg.tool_call_id == ""
    assert "unknown_tool" in cast("str", msg.content)
