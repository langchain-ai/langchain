"""Unit tests for invalid_tool_calls handling in create_agent.

These tests verify that when an LLM returns invalid_tool_calls (e.g., from
JSON parsing errors), the agent correctly converts them to error ToolMessages
and routes back to the model for retry instead of silently exiting.

Fixes issue #33504.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain.agents.factory import (
    _fetch_last_ai_and_tool_messages,
    _make_model_to_tools_edge,
)


def test_routing_with_invalid_tool_calls_routes_to_model() -> None:
    """When tool_calls is empty but invalid_tool_calls exists, route to model."""
    model_to_tools = _make_model_to_tools_edge(
        model_destination="model",
        structured_output_tools={},
        end_destination="__end__",
    )

    ai_msg = AIMessage(
        content="",
        tool_calls=[],
        invalid_tool_calls=[
            {
                "id": "call-123",
                "name": "write_file",
                "args": '{"file_path": "test.txt", "content": "Hello"]',
                "error": "Expecting ',' delimiter",
                "type": "invalid_tool_call",
            }
        ],
    )
    error_tool_msg = ToolMessage(
        content="Error: Failed to parse tool call arguments.",
        tool_call_id="call-123",
        status="error",
    )
    state = {"messages": [HumanMessage(content="test"), ai_msg, error_tool_msg]}

    result = model_to_tools(state)
    assert result == "model", (
        "Expected routing to model for retry when invalid_tool_calls exist, "
        f"but got {result!r}"
    )


def test_routing_without_invalid_tool_calls_exits() -> None:
    """When tool_calls is empty and no invalid_tool_calls, route to end."""
    model_to_tools = _make_model_to_tools_edge(
        model_destination="model",
        structured_output_tools={},
        end_destination="__end__",
    )

    ai_msg = AIMessage(content="Done!", tool_calls=[])
    state = {"messages": [HumanMessage(content="test"), ai_msg]}

    result = model_to_tools(state)
    assert result == "__end__", (
        f"Expected routing to __end__ when no tool calls, but got {result!r}"
    )


def test_routing_with_empty_invalid_tool_calls_exits() -> None:
    """When both tool_calls and invalid_tool_calls are empty, route to end."""
    model_to_tools = _make_model_to_tools_edge(
        model_destination="model",
        structured_output_tools={},
        end_destination="__end__",
    )

    ai_msg = AIMessage(content="Done!", tool_calls=[], invalid_tool_calls=[])
    state = {"messages": [HumanMessage(content="test"), ai_msg]}

    result = model_to_tools(state)
    assert result == "__end__", (
        f"Expected routing to __end__ when invalid_tool_calls is empty, but got {result!r}"
    )
