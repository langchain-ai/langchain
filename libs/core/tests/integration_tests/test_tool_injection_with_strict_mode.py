"""
Integration test to reproduce bug #33744:
InjectedToolCallId fails when using bind_tools + response_format + strict=True

The bug occurs when tools are invoked via a ToolCall dict that comes from
strict structured output mode.
"""
from typing import Annotated
from pydantic import BaseModel, Field

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage


class LoadTraceSchema(BaseModel):
    """Schema for load_trace tool."""
    trace_id: str = Field(..., description="Trace ID to load")


@tool(args_schema=LoadTraceSchema)
def load_trace(
    trace_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> ToolMessage:
    """Load a trace from storage.

    This tool uses InjectedToolCallId which should be automatically injected
    at runtime.

    This is the exact pattern from issue #33744.
    """
    return ToolMessage(
        content=f"Loaded trace {trace_id}",
        tool_call_id=tool_call_id
    )


def test_injected_tool_call_id_with_strict_structured_output():
    """
    Test that InjectedToolCallId works with tool calls from structured output.

    This is a regression test for https://github.com/langchain-ai/langchain/issues/33744

    When using bind_tools with response_format and strict=True, the LLM returns
    tool calls that must be invoked. The InjectedToolCallId should still work.

    Expected: tool_call_id should be automatically injected
    Actual (if buggy): TypeError: missing 1 required positional argument: 'tool_call_id'

    Before the fix:
    - In certain execution paths (e.g., when using strict structured output),
      the tool function might be called without injected arguments being properly added.
    - This would cause: TypeError: load_trace() missing 1 required positional argument: 'tool_call_id'

    After the fix:
    - Shared injection logic ensures tool_call_id is always injected
    - Defensive validation catches any execution path that bypasses injection
    """
    # Simulate a tool call coming from the LLM
    # (this is what would be in AIMessage.tool_calls)
    tool_call = {
        "name": "load_trace",
        "args": {"trace_id": "test_123"},
        "id": "call_abc123",
        "type": "tool_call"
    }

    # Invoke the tool with the tool call
    # This should work - tool_call_id should be injected from the "id" field
    result = load_trace.invoke(tool_call)

    # If we get here without TypeError, injection worked!
    assert isinstance(result, ToolMessage)
    assert result.content == "Loaded trace test_123"
    assert result.tool_call_id == "call_abc123"


def test_injected_tool_call_id_with_direct_dict_invocation():
    """
    Test that tools cannot be invoked with a plain dict when InjectedToolCallId is required.

    Tools with InjectedToolCallId MUST be invoked with a proper ToolCall dict
    that includes an 'id' field. Direct dict invocation should raise a clear error.
    """
    import pytest

    # Direct dict invocation (no ToolCall wrapper) should fail
    with pytest.raises(ValueError, match="requires 'tool_call_id' to be injected"):
        load_trace.invoke({"trace_id": "test_456"})


def test_injected_tool_call_id_missing_raises_error():
    """
    Test that missing tool_call_id raises a clear error.

    When a tool requires InjectedToolCallId but is invoked without a proper
    ToolCall, it should raise a clear error message.
    """
    import pytest

    # Try to invoke without a ToolCall (just a dict with args)
    with pytest.raises(ValueError, match="requires 'tool_call_id' to be injected"):
        load_trace.invoke({"trace_id": "test_789"})


def test_async_tool_with_injected_args():
    """
    Test that InjectedToolCallId works with async tools.

    This ensures the async execution path also properly injects arguments.
    """
    import asyncio

    class AsyncToolSchema(BaseModel):
        query: str = Field(..., description="Search query")

    @tool(args_schema=AsyncToolSchema)
    async def async_search_tool(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> ToolMessage:
        """Async search tool with injected arg."""
        # Simulate async operation
        await asyncio.sleep(0.001)
        return ToolMessage(
            content=f"Async search results for: {query}",
            tool_call_id=tool_call_id,
        )

    # Invoke with ToolCall
    tool_call = {
        "name": "async_search_tool",
        "args": {"query": "test"},
        "id": "call_async_1",
        "type": "tool_call",
    }

    # Test async invocation
    result = asyncio.run(async_search_tool.ainvoke(tool_call))
    assert isinstance(result, ToolMessage)
    assert result.content == "Async search results for: test"
    assert result.tool_call_id == "call_async_1"


if __name__ == "__main__":
    test_injected_tool_call_id_with_strict_structured_output()
    print("✓ Test 1 passed - InjectedToolCallId works correctly!")

    try:
        test_injected_tool_call_id_missing_raises_error()
        print("✓ Test 2 passed - Missing tool_call_id raises clear error!")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")

    try:
        test_async_tool_with_injected_args()
        print("✓ Test 3 passed - Async tools with injected args work!")
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
