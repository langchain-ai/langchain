import re
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.runtime import Runtime

from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.human_in_the_loop import (
    Action,
    HumanInTheLoopMiddleware,
    RejectionResponseFactory,
)
from langchain.agents.middleware.types import AgentState


def test_human_in_the_loop_middleware_initialization() -> None:
    """Test HumanInTheLoopMiddleware initialization."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}},
        description_prefix="Custom prefix",
    )

    assert middleware.interrupt_on == {
        "test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}
    }
    assert middleware.description_prefix == "Custom prefix"


def test_human_in_the_loop_middleware_no_interrupts_needed() -> None:
    """Test HumanInTheLoopMiddleware when no interrupts are needed."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    # Test with no messages
    state = AgentState[Any](messages=[])
    result = middleware.after_model(state, Runtime())
    assert result is None

    # Test with message but no tool calls
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), AIMessage(content="Hi there")])

    result = middleware.after_model(state, Runtime())
    assert result is None

    # Test with tool calls that don't require interrupts
    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "other_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])
    result = middleware.after_model(state, Runtime())
    assert result is None


def test_human_in_the_loop_middleware_single_tool_accept() -> None:
    """Test HumanInTheLoopMiddleware with single tool accept response."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_accept(_: Any) -> dict[str, Any]:
        return {"decisions": [{"type": "approve"}]}

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_accept):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0] == ai_message
        assert result["messages"][0].tool_calls == ai_message.tool_calls

    state["messages"].append(
        ToolMessage(content="Tool message", name="test_tool", tool_call_id="1")
    )
    state["messages"].append(AIMessage(content="test_tool called with result: Tool message"))

    result = middleware.after_model(state, Runtime())
    # No interrupts needed
    assert result is None


def test_human_in_the_loop_middleware_single_tool_edit() -> None:
    """Test HumanInTheLoopMiddleware with single tool edit response."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_edit(_: Any) -> dict[str, Any]:
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="test_tool",
                        args={"input": "edited"},
                    ),
                }
            ]
        }

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_edit):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls[0]["args"] == {"input": "edited"}
        assert result["messages"][0].tool_calls[0]["id"] == "1"  # ID should be preserved


def test_human_in_the_loop_middleware_single_tool_response() -> None:
    """Test HumanInTheLoopMiddleware with single tool response with custom message."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_response(_: Any) -> dict[str, Any]:
        return {"decisions": [{"type": "reject", "message": "Custom response message"}]}

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_response
    ):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], AIMessage)
        assert isinstance(result["messages"][1], ToolMessage)
        assert result["messages"][1].content == "Custom response message"
        assert result["messages"][1].name == "test_tool"
        assert result["messages"][1].tool_call_id == "1"
        assert result["messages"][1].status == "error"


def test_human_in_the_loop_middleware_single_tool_respond() -> None:
    """Test HumanInTheLoopMiddleware with `respond` decision producing a success ToolMessage."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"ask_user": {"allowed_decisions": ["respond"]}}
    )

    ai_message = AIMessage(
        content="Let me ask the user.",
        tool_calls=[{"name": "ask_user", "args": {"question": "favorite color?"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_respond(_: Any) -> dict[str, Any]:
        return {"decisions": [{"type": "respond", "message": "blue"}]}

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_respond):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], AIMessage)
        # Tool call is preserved on the AI message (provider APIs require pairing).
        assert len(result["messages"][0].tool_calls) == 1
        assert result["messages"][0].tool_calls[0]["id"] == "1"

        tool_message = result["messages"][1]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.content == "blue"
        assert tool_message.name == "ask_user"
        assert tool_message.tool_call_id == "1"
        assert tool_message.status == "success"


def test_human_in_the_loop_middleware_respond_disallowed() -> None:
    """Test that `respond` raises when not in `allowed_decisions`."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_respond(_: Any) -> dict[str, Any]:
        return {"decisions": [{"type": "respond", "message": "synthetic"}]}

    with (
        patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_respond),
        pytest.raises(
            ValueError,
            match=re.escape(
                "Decision type 'respond' is not allowed for tool 'test_tool'. "
                "Expected one of ['approve', 'edit', 'reject'] based on the tool's "
                "configuration."
            ),
        ),
    ):
        middleware.after_model(state, Runtime())


def test_human_in_the_loop_middleware_mixed_with_respond() -> None:
    """Test mixed decisions: one tool approved, one tool answered via `respond`."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "get_forecast": {"allowed_decisions": ["approve"]},
            "ask_user": {"allowed_decisions": ["respond"]},
        }
    )

    ai_message = AIMessage(
        content="Two things",
        tool_calls=[
            {"name": "get_forecast", "args": {"location": "SF"}, "id": "1"},
            {"name": "ask_user", "args": {"question": "favorite color?"}, "id": "2"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hi"), ai_message])

    def mock_mixed(_: Any) -> dict[str, Any]:
        return {
            "decisions": [
                {"type": "approve"},
                {"type": "respond", "message": "blue"},
            ]
        }

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_mixed):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        # AI message + 1 synthetic ToolMessage for the respond decision.
        assert len(result["messages"]) == 2

        updated_ai_message = result["messages"][0]
        assert len(updated_ai_message.tool_calls) == 2
        assert updated_ai_message.tool_calls[0]["name"] == "get_forecast"
        assert updated_ai_message.tool_calls[1]["name"] == "ask_user"

        tool_message = result["messages"][1]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.content == "blue"
        assert tool_message.name == "ask_user"
        assert tool_message.tool_call_id == "2"
        assert tool_message.status == "success"


def test_human_in_the_loop_middleware_true_allows_respond() -> None:
    """Test that the `True` shortcut permits `respond` decisions."""
    middleware = HumanInTheLoopMiddleware(interrupt_on={"ask_user": True})

    ai_message = AIMessage(
        content="Asking",
        tool_calls=[{"name": "ask_user", "args": {"q": "?"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hi"), ai_message])

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "respond", "message": "answer"}]},
    ):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert len(result["messages"]) == 2
        tool_message = result["messages"][1]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.content == "answer"
        assert tool_message.status == "success"


def test_human_in_the_loop_middleware_multiple_tools_mixed_responses() -> None:
    """Test HumanInTheLoopMiddleware with multiple tools and mixed response types."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "get_forecast": {"allowed_decisions": ["approve", "edit", "reject"]},
            "get_temperature": {"allowed_decisions": ["approve", "edit", "reject"]},
        }
    )

    ai_message = AIMessage(
        content="I'll help you with weather",
        tool_calls=[
            {"name": "get_forecast", "args": {"location": "San Francisco"}, "id": "1"},
            {"name": "get_temperature", "args": {"location": "San Francisco"}, "id": "2"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="What's the weather?"), ai_message])

    def mock_mixed_responses(_: Any) -> dict[str, Any]:
        return {
            "decisions": [
                {"type": "approve"},
                {"type": "reject", "message": "User rejected this tool call"},
            ]
        }

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_mixed_responses
    ):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        assert (
            len(result["messages"]) == 2
        )  # AI message with accepted tool call + tool message for rejected

        # First message should be the AI message with both tool calls
        updated_ai_message = result["messages"][0]
        assert len(updated_ai_message.tool_calls) == 2  # Both tool calls remain
        assert updated_ai_message.tool_calls[0]["name"] == "get_forecast"  # Accepted
        assert updated_ai_message.tool_calls[1]["name"] == "get_temperature"  # Got response

        # Second message should be the tool message for the rejected tool call
        tool_message = result["messages"][1]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.content == "User rejected this tool call"
        assert tool_message.name == "get_temperature"
        assert tool_message.status == "error"


def test_human_in_the_loop_middleware_multiple_tools_edit_responses() -> None:
    """Test HumanInTheLoopMiddleware with multiple tools and edit responses."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "get_forecast": {"allowed_decisions": ["approve", "edit", "reject"]},
            "get_temperature": {"allowed_decisions": ["approve", "edit", "reject"]},
        }
    )

    ai_message = AIMessage(
        content="I'll help you with weather",
        tool_calls=[
            {"name": "get_forecast", "args": {"location": "San Francisco"}, "id": "1"},
            {"name": "get_temperature", "args": {"location": "San Francisco"}, "id": "2"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="What's the weather?"), ai_message])

    def mock_edit_responses(_: Any) -> dict[str, Any]:
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="get_forecast",
                        args={"location": "New York"},
                    ),
                },
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="get_temperature",
                        args={"location": "New York"},
                    ),
                },
            ]
        }

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_edit_responses
    ):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1

        updated_ai_message = result["messages"][0]
        assert updated_ai_message.tool_calls[0]["args"] == {"location": "New York"}
        assert updated_ai_message.tool_calls[0]["id"] == "1"  # ID preserved
        assert updated_ai_message.tool_calls[1]["args"] == {"location": "New York"}
        assert updated_ai_message.tool_calls[1]["id"] == "2"  # ID preserved


def test_human_in_the_loop_middleware_edit_with_modified_args() -> None:
    """Test HumanInTheLoopMiddleware with edit action that includes modified args."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_edit_with_args(_: Any) -> dict[str, Any]:
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="test_tool",
                        args={"input": "modified"},
                    ),
                }
            ]
        }

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        side_effect=mock_edit_with_args,
    ):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1

        # Should have modified args
        updated_ai_message = result["messages"][0]
        assert updated_ai_message.tool_calls[0]["args"] == {"input": "modified"}
        assert updated_ai_message.tool_calls[0]["id"] == "1"  # ID preserved


def test_human_in_the_loop_middleware_unknown_response_type() -> None:
    """Test HumanInTheLoopMiddleware with unknown response type."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_unknown(_: Any) -> dict[str, Any]:
        return {"decisions": [{"type": "unknown"}]}

    with (
        patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_unknown),
        pytest.raises(
            ValueError,
            match=re.escape(
                "Unexpected human decision: {'type': 'unknown'}. "
                "Decision type 'unknown' is not allowed for tool 'test_tool'. "
                "Expected one of ['approve', 'edit', 'reject'] based on the tool's "
                "configuration."
            ),
        ),
    ):
        middleware.after_model(state, Runtime())


def test_human_in_the_loop_middleware_disallowed_action() -> None:
    """Test HumanInTheLoopMiddleware with action not allowed by tool config."""
    # edit is not allowed by tool config
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_disallowed_action(_: Any) -> dict[str, Any]:
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="test_tool",
                        args={"input": "modified"},
                    ),
                }
            ]
        }

    with (
        patch(
            "langchain.agents.middleware.human_in_the_loop.interrupt",
            side_effect=mock_disallowed_action,
        ),
        pytest.raises(
            ValueError,
            match=re.escape(
                "Unexpected human decision: {'type': 'edit', 'edited_action': "
                "{'name': 'test_tool', 'args': {'input': 'modified'}}}. "
                "Decision type 'edit' is not allowed for tool 'test_tool'. "
                "Expected one of ['approve', 'reject'] based on the tool's "
                "configuration."
            ),
        ),
    ):
        middleware.after_model(state, Runtime())


def test_human_in_the_loop_middleware_mixed_auto_approved_and_interrupt() -> None:
    """Test HumanInTheLoopMiddleware with mix of auto-approved and interrupt tools."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"interrupt_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[
            {"name": "auto_tool", "args": {"input": "auto"}, "id": "1"},
            {"name": "interrupt_tool", "args": {"input": "interrupt"}, "id": "2"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_accept(_: Any) -> dict[str, Any]:
        return {"decisions": [{"type": "approve"}]}

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_accept):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1

        updated_ai_message = result["messages"][0]
        # Should have both tools: auto-approved first, then interrupt tool
        assert len(updated_ai_message.tool_calls) == 2
        assert updated_ai_message.tool_calls[0]["name"] == "auto_tool"
        assert updated_ai_message.tool_calls[1]["name"] == "interrupt_tool"


def test_human_in_the_loop_middleware_interrupt_request_structure() -> None:
    """Test that interrupt requests are structured correctly."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}},
        description_prefix="Custom prefix",
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test", "location": "SF"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    captured_request = None

    def mock_capture_requests(request: Any) -> dict[str, Any]:
        nonlocal captured_request
        captured_request = request
        return {"decisions": [{"type": "approve"}]}

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_capture_requests
    ):
        middleware.after_model(state, Runtime())

        assert captured_request is not None
        assert "action_requests" in captured_request
        assert "review_configs" in captured_request

        assert len(captured_request["action_requests"]) == 1
        action_request = captured_request["action_requests"][0]
        assert action_request["name"] == "test_tool"
        assert action_request["args"] == {"input": "test", "location": "SF"}
        assert "Custom prefix" in action_request["description"]
        assert "Tool: test_tool" in action_request["description"]
        assert "Args: {'input': 'test', 'location': 'SF'}" in action_request["description"]

        assert len(captured_request["review_configs"]) == 1
        review_config = captured_request["review_configs"][0]
        assert review_config["action_name"] == "test_tool"
        assert review_config["allowed_decisions"] == ["approve", "edit", "reject"]


def test_human_in_the_loop_middleware_boolean_configs() -> None:
    """Test HITL middleware with boolean tool configs."""
    middleware = HumanInTheLoopMiddleware(interrupt_on={"test_tool": True})

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    # Test accept
    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "approve"}]},
    ):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls == ai_message.tool_calls

    # Test edit
    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="test_tool",
                        args={"input": "edited"},
                    ),
                }
            ]
        },
    ):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls[0]["args"] == {"input": "edited"}

    middleware = HumanInTheLoopMiddleware(interrupt_on={"test_tool": False})

    result = middleware.after_model(state, Runtime())
    # No interruption should occur
    assert result is None


def test_human_in_the_loop_middleware_sequence_mismatch() -> None:
    """Test that sequence mismatch in resume raises an error."""
    middleware = HumanInTheLoopMiddleware(interrupt_on={"test_tool": True})

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    # Test with too few responses
    with (
        patch(
            "langchain.agents.middleware.human_in_the_loop.interrupt",
            return_value={"decisions": []},  # No responses for 1 tool call
        ),
        pytest.raises(
            ValueError,
            match=re.escape(
                "Number of human decisions (0) does not match number of hanging tool calls (1)."
            ),
        ),
    ):
        middleware.after_model(state, Runtime())

    # Test with too many responses
    with (
        patch(
            "langchain.agents.middleware.human_in_the_loop.interrupt",
            return_value={
                "decisions": [
                    {"type": "approve"},
                    {"type": "approve"},
                ]
            },  # 2 responses for 1 tool call
        ),
        pytest.raises(
            ValueError,
            match=re.escape(
                "Number of human decisions (2) does not match number of hanging tool calls (1)."
            ),
        ),
    ):
        middleware.after_model(state, Runtime())


def test_human_in_the_loop_middleware_description_as_callable() -> None:
    """Test that description field accepts both string and callable."""

    def custom_description(
        tool_call: ToolCall, state: AgentState[Any], runtime: Runtime[None]
    ) -> str:
        """Generate a custom description."""
        return f"Custom: {tool_call['name']} with args {tool_call['args']}"

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "tool_with_callable": InterruptOnConfig(
                allowed_decisions=["approve"],
                description=custom_description,
            ),
            "tool_with_string": InterruptOnConfig(
                allowed_decisions=["approve"],
                description="Static description",
            ),
        }
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[
            {"name": "tool_with_callable", "args": {"x": 1}, "id": "1"},
            {"name": "tool_with_string", "args": {"y": 2}, "id": "2"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    captured_request = None

    def mock_capture_requests(request: Any) -> dict[str, Any]:
        nonlocal captured_request
        captured_request = request
        return {"decisions": [{"type": "approve"}, {"type": "approve"}]}

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_capture_requests
    ):
        middleware.after_model(state, Runtime())

        assert captured_request is not None
        assert "action_requests" in captured_request
        assert len(captured_request["action_requests"]) == 2

        # Check callable description
        assert (
            captured_request["action_requests"][0]["description"]
            == "Custom: tool_with_callable with args {'x': 1}"
        )

        # Check string description
        assert captured_request["action_requests"][1]["description"] == "Static description"


def test_human_in_the_loop_middleware_preserves_tool_call_order() -> None:
    """Test that middleware preserves the original order of tool calls.

    This test verifies that when mixing auto-approved and interrupt tools,
    the final tool call order matches the original order from the AI message.
    """
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "tool_b": {"allowed_decisions": ["approve", "edit", "reject"]},
            "tool_d": {"allowed_decisions": ["approve", "edit", "reject"]},
        }
    )

    # Create AI message with interleaved auto-approved and interrupt tools
    # Order: auto (A) -> interrupt (B) -> auto (C) -> interrupt (D) -> auto (E)
    ai_message = AIMessage(
        content="Processing multiple tools",
        tool_calls=[
            {"name": "tool_a", "args": {"val": 1}, "id": "id_a"},
            {"name": "tool_b", "args": {"val": 2}, "id": "id_b"},
            {"name": "tool_c", "args": {"val": 3}, "id": "id_c"},
            {"name": "tool_d", "args": {"val": 4}, "id": "id_d"},
            {"name": "tool_e", "args": {"val": 5}, "id": "id_e"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_approve_all(_: Any) -> dict[str, Any]:
        # Approve both interrupt tools (B and D)
        return {"decisions": [{"type": "approve"}, {"type": "approve"}]}

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_approve_all
    ):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result

        updated_ai_message = result["messages"][0]
        assert len(updated_ai_message.tool_calls) == 5

        # Verify original order is preserved: A -> B -> C -> D -> E
        assert updated_ai_message.tool_calls[0]["name"] == "tool_a"
        assert updated_ai_message.tool_calls[0]["id"] == "id_a"
        assert updated_ai_message.tool_calls[1]["name"] == "tool_b"
        assert updated_ai_message.tool_calls[1]["id"] == "id_b"
        assert updated_ai_message.tool_calls[2]["name"] == "tool_c"
        assert updated_ai_message.tool_calls[2]["id"] == "id_c"
        assert updated_ai_message.tool_calls[3]["name"] == "tool_d"
        assert updated_ai_message.tool_calls[3]["id"] == "id_d"
        assert updated_ai_message.tool_calls[4]["name"] == "tool_e"
        assert updated_ai_message.tool_calls[4]["id"] == "id_e"


def test_human_in_the_loop_middleware_preserves_order_with_edits() -> None:
    """Test that order is preserved when interrupt tools are edited."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "tool_b": {"allowed_decisions": ["approve", "edit", "reject"]},
            "tool_d": {"allowed_decisions": ["approve", "edit", "reject"]},
        }
    )

    ai_message = AIMessage(
        content="Processing multiple tools",
        tool_calls=[
            {"name": "tool_a", "args": {"val": 1}, "id": "id_a"},
            {"name": "tool_b", "args": {"val": 2}, "id": "id_b"},
            {"name": "tool_c", "args": {"val": 3}, "id": "id_c"},
            {"name": "tool_d", "args": {"val": 4}, "id": "id_d"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_edit_responses(_: Any) -> dict[str, Any]:
        # Edit tool_b, approve tool_d
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(name="tool_b", args={"val": 200}),
                },
                {"type": "approve"},
            ]
        }

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_edit_responses
    ):
        result = middleware.after_model(state, Runtime())
        assert result is not None

        updated_ai_message = result["messages"][0]
        assert len(updated_ai_message.tool_calls) == 4

        # Verify order: A (auto) -> B (edited) -> C (auto) -> D (approved)
        assert updated_ai_message.tool_calls[0]["name"] == "tool_a"
        assert updated_ai_message.tool_calls[0]["args"] == {"val": 1}
        assert updated_ai_message.tool_calls[1]["name"] == "tool_b"
        assert updated_ai_message.tool_calls[1]["args"] == {"val": 200}  # Edited
        assert updated_ai_message.tool_calls[1]["id"] == "id_b"  # ID preserved
        assert updated_ai_message.tool_calls[2]["name"] == "tool_c"
        assert updated_ai_message.tool_calls[2]["args"] == {"val": 3}
        assert updated_ai_message.tool_calls[3]["name"] == "tool_d"
        assert updated_ai_message.tool_calls[3]["args"] == {"val": 4}


def test_human_in_the_loop_middleware_preserves_order_with_rejections() -> None:
    """Test that order is preserved when some interrupt tools are rejected."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "tool_b": {"allowed_decisions": ["approve", "edit", "reject"]},
            "tool_d": {"allowed_decisions": ["approve", "edit", "reject"]},
        }
    )

    ai_message = AIMessage(
        content="Processing multiple tools",
        tool_calls=[
            {"name": "tool_a", "args": {"val": 1}, "id": "id_a"},
            {"name": "tool_b", "args": {"val": 2}, "id": "id_b"},
            {"name": "tool_c", "args": {"val": 3}, "id": "id_c"},
            {"name": "tool_d", "args": {"val": 4}, "id": "id_d"},
            {"name": "tool_e", "args": {"val": 5}, "id": "id_e"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_mixed_responses(_: Any) -> dict[str, Any]:
        # Reject tool_b, approve tool_d
        return {
            "decisions": [
                {"type": "reject", "message": "Rejected tool B"},
                {"type": "approve"},
            ]
        }

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_mixed_responses
    ):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert len(result["messages"]) == 2  # AI message + tool message for rejection

        updated_ai_message = result["messages"][0]
        # tool_b is still in the list (with rejection handled via tool message)
        assert len(updated_ai_message.tool_calls) == 5

        # Verify order maintained: A (auto) -> B (rejected) -> C (auto) -> D (approved) -> E (auto)
        assert updated_ai_message.tool_calls[0]["name"] == "tool_a"
        assert updated_ai_message.tool_calls[1]["name"] == "tool_b"
        assert updated_ai_message.tool_calls[2]["name"] == "tool_c"
        assert updated_ai_message.tool_calls[3]["name"] == "tool_d"
        assert updated_ai_message.tool_calls[4]["name"] == "tool_e"

        # Check rejection tool message
        tool_message = result["messages"][1]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.content == "Rejected tool B"
        assert tool_message.tool_call_id == "id_b"
        assert tool_message.status == "error"


def test_human_in_the_loop_middleware_reject_default_status_and_content() -> None:
    """The default rejection message uses `status="error"` and the canned content.

    Pinned so that any future flip of the default is intentional and visible
    rather than a silent wire-format change.
    """
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help",
        tool_calls=[{"name": "test_tool", "args": {"input": "x"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "reject"}]},
    ):
        result = middleware.after_model(state, Runtime())

    assert result is not None
    tool_message = result["messages"][1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "error"
    assert tool_message.content == "User rejected the tool call for `test_tool` with id 1"
    assert tool_message.tool_call_id == "1"
    assert tool_message.name == "test_tool"


def test_human_in_the_loop_middleware_reject_uses_human_message() -> None:
    """A user-supplied `message` on the reject decision becomes the ToolMessage content."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help",
        tool_calls=[{"name": "test_tool", "args": {"input": "x"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "reject", "message": "no thanks"}]},
    ):
        result = middleware.after_model(state, Runtime())

    assert result is not None
    tool_message = result["messages"][1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "no thanks"
    assert tool_message.status == "error"


def test_human_in_the_loop_middleware_rejection_response_callable() -> None:
    """`rejection_response` callable receives the tool call and decision."""
    captured: dict[str, Any] = {}

    def factory(tool_call: ToolCall, decision: dict[str, Any]) -> ToolMessage:
        captured["tool_call"] = tool_call
        captured["decision"] = decision
        return ToolMessage(
            content=f"declined: {decision.get('message', '?')}",
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status="success",
            additional_kwargs={"hitl_decision": "reject"},
        )

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {
                "allowed_decisions": ["reject"],
                "rejection_response": factory,
            }
        }
    )

    ai_message = AIMessage(
        content="I'll help",
        tool_calls=[{"name": "test_tool", "args": {"input": "x"}, "id": "tc-1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "reject", "message": "stop"}]},
    ):
        result = middleware.after_model(state, Runtime())

    assert result is not None
    tool_message = result["messages"][1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "declined: stop"
    assert tool_message.additional_kwargs == {"hitl_decision": "reject"}
    assert captured["tool_call"]["id"] == "tc-1"
    assert captured["decision"] == {"type": "reject", "message": "stop"}


def test_human_in_the_loop_middleware_rejection_response_static_toolmessage() -> None:
    """A static `ToolMessage` override is used verbatim with id/name stamped from the call."""
    template = ToolMessage(
        content="Rejected by policy.",
        name="placeholder",
        tool_call_id="placeholder",
        status="error",
    )

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {
                "allowed_decisions": ["reject"],
                "rejection_response": template,
            }
        }
    )

    ai_message = AIMessage(
        content="I'll help",
        tool_calls=[{"name": "test_tool", "args": {"input": "x"}, "id": "tc-1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "reject"}]},
    ):
        result = middleware.after_model(state, Runtime())

    assert result is not None
    tool_message = result["messages"][1]
    assert isinstance(tool_message, ToolMessage)
    # Content/status preserved from the template; identity fields stamped from the call.
    assert tool_message.content == "Rejected by policy."
    assert tool_message.status == "error"
    assert tool_message.name == "test_tool"
    assert tool_message.tool_call_id == "tc-1"
    # Template instance must not be mutated — callers may share one across tools.
    assert template.name == "placeholder"
    assert template.tool_call_id == "placeholder"


def test_human_in_the_loop_middleware_rejection_response_callable_id_mismatch() -> None:
    """A callable returning a `ToolMessage` with the wrong `tool_call_id` raises.

    Providers (notably Anthropic) reject AI/Tool message pairs whose ids do not
    line up, so the middleware fails fast rather than letting the malformed
    pair reach the model.
    """

    def bad_factory(tool_call: ToolCall, decision: dict[str, Any]) -> ToolMessage:
        return ToolMessage(
            content="declined",
            name=tool_call["name"],
            tool_call_id="not-the-right-id",
            status="success",
        )

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {
                "allowed_decisions": ["reject"],
                "rejection_response": bad_factory,
            }
        }
    )

    ai_message = AIMessage(
        content="I'll help",
        tool_calls=[{"name": "test_tool", "args": {"input": "x"}, "id": "tc-1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    with (
        patch(
            "langchain.agents.middleware.human_in_the_loop.interrupt",
            return_value={"decisions": [{"type": "reject"}]},
        ),
        pytest.raises(
            ValueError,
            match=re.escape(
                "`rejection_response` callable for tool 'test_tool' returned a "
                "`ToolMessage` with `tool_call_id='not-the-right-id'`, but the "
                "rejected call's id is 'tc-1'."
            ),
        ),
    ):
        middleware.after_model(state, Runtime())


def test_human_in_the_loop_middleware_reject_empty_message_falls_back_to_default() -> None:
    """An empty `decision["message"]` falls back to the default canned content."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help",
        tool_calls=[{"name": "test_tool", "args": {"input": "x"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "reject", "message": ""}]},
    ):
        result = middleware.after_model(state, Runtime())

    assert result is not None
    tool_message = result["messages"][1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "User rejected the tool call for `test_tool` with id 1"
    assert tool_message.status == "error"


def test_human_in_the_loop_middleware_per_tool_rejection_response() -> None:
    """Each rejected tool call resolves its own `rejection_response` from the per-tool config.

    Guards against a regression in `_process_decision` that misroutes the
    override (e.g. by reading `rejection_response` from the wrong entry in
    `interrupt_on` or by sharing it across calls).
    """

    def factory_a(tool_call: ToolCall, decision: dict[str, Any]) -> ToolMessage:
        return ToolMessage(
            content="A declined",
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status="success",
        )

    template_b = ToolMessage(
        content="B declined by policy.",
        name="placeholder",
        tool_call_id="placeholder",
        status="error",
    )

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "tool_a": {"allowed_decisions": ["reject"], "rejection_response": factory_a},
            "tool_b": {"allowed_decisions": ["reject"], "rejection_response": template_b},
        }
    )

    ai_message = AIMessage(
        content="Two calls",
        tool_calls=[
            {"name": "tool_a", "args": {}, "id": "id_a"},
            {"name": "tool_b", "args": {}, "id": "id_b"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hi"), ai_message])

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "reject"}, {"type": "reject"}]},
    ):
        result = middleware.after_model(state, Runtime())

    assert result is not None
    # AI message + 2 synthetic ToolMessages, one per rejected call.
    assert len(result["messages"]) == 3

    msg_a = result["messages"][1]
    assert isinstance(msg_a, ToolMessage)
    assert msg_a.content == "A declined"
    assert msg_a.tool_call_id == "id_a"
    assert msg_a.name == "tool_a"
    assert msg_a.status == "success"

    msg_b = result["messages"][2]
    assert isinstance(msg_b, ToolMessage)
    assert msg_b.content == "B declined by policy."
    assert msg_b.tool_call_id == "id_b"
    assert msg_b.name == "tool_b"
    assert msg_b.status == "error"


def test_human_in_the_loop_middleware_mixed_approve_and_reject_callable() -> None:
    """An approve+reject pair routes the rejection through its `rejection_response` callable."""

    def factory(tool_call: ToolCall, decision: dict[str, Any]) -> ToolMessage:
        return ToolMessage(
            content="declined",
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status="success",
            additional_kwargs={"hitl_decision": "reject"},
        )

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "approve_tool": {"allowed_decisions": ["approve"]},
            "reject_tool": {
                "allowed_decisions": ["reject"],
                "rejection_response": factory,
            },
        }
    )

    ai_message = AIMessage(
        content="Two calls",
        tool_calls=[
            {"name": "approve_tool", "args": {}, "id": "id_ok"},
            {"name": "reject_tool", "args": {}, "id": "id_no"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hi"), ai_message])

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "approve"}, {"type": "reject"}]},
    ):
        result = middleware.after_model(state, Runtime())

    assert result is not None
    # AI message + 1 synthetic ToolMessage for the reject decision.
    assert len(result["messages"]) == 2

    updated_ai = result["messages"][0]
    assert len(updated_ai.tool_calls) == 2
    assert updated_ai.tool_calls[0]["name"] == "approve_tool"
    assert updated_ai.tool_calls[1]["name"] == "reject_tool"

    tool_message = result["messages"][1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "declined"
    assert tool_message.tool_call_id == "id_no"
    assert tool_message.additional_kwargs == {"hitl_decision": "reject"}


def test_rejection_response_factory_alias_is_callable_signature() -> None:
    """`RejectionResponseFactory` is a public alias users can annotate factories against."""

    def my_factory(tool_call: ToolCall, decision: dict[str, Any]) -> ToolMessage:
        return ToolMessage(
            content="ok",
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status="success",
        )

    factory: RejectionResponseFactory = my_factory
    msg = factory(ToolCall(name="t", args={}, id="x"), {"type": "reject"})
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "x"


def test_human_in_the_loop_middleware_rejection_response_template_status_preserved() -> None:
    """A template's non-default `status` survives the `model_copy` stamp.

    The documented use case for overriding is `status="success"` to suppress
    retry loops on providers that re-emit on `status="error"`. Pin that the
    copy only overwrites identity fields, not `status` or content.
    """
    template = ToolMessage(
        content="Declined; do not retry.",
        name="placeholder",
        tool_call_id="placeholder",
        status="success",
        additional_kwargs={"hitl_decision": "reject"},
    )

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {
                "allowed_decisions": ["reject"],
                "rejection_response": template,
            }
        }
    )

    ai_message = AIMessage(
        content="I'll help",
        tool_calls=[{"name": "test_tool", "args": {}, "id": "tc-1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "reject"}]},
    ):
        result = middleware.after_model(state, Runtime())

    assert result is not None
    tool_message = result["messages"][1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "success"
    assert tool_message.content == "Declined; do not retry."
    assert tool_message.additional_kwargs == {"hitl_decision": "reject"}
    assert tool_message.tool_call_id == "tc-1"
    assert tool_message.name == "test_tool"


def test_human_in_the_loop_middleware_heterogeneous_batch_routes_per_tool() -> None:
    """A mixed approve/edit/respond/reject batch routes each decision to its tool's config.

    Higher-leverage than a homogeneous reject batch: regressions in
    `_process_decision` (e.g. reading `rejection_response` from the wrong
    `interrupt_on` entry) only surface when decision *types* differ across
    calls, not just tools.
    """

    def reject_factory(tool_call: ToolCall, decision: dict[str, Any]) -> ToolMessage:
        return ToolMessage(
            content="declined by factory",
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status="success",
        )

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "edit_tool": {"allowed_decisions": ["edit"]},
            "respond_tool": {"allowed_decisions": ["respond"]},
            "reject_tool": {
                "allowed_decisions": ["reject"],
                "rejection_response": reject_factory,
            },
        }
    )

    ai_message = AIMessage(
        content="Three calls",
        tool_calls=[
            {"name": "edit_tool", "args": {"x": 1}, "id": "id_edit"},
            {"name": "respond_tool", "args": {}, "id": "id_resp"},
            {"name": "reject_tool", "args": {}, "id": "id_rej"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hi"), ai_message])

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {"name": "edit_tool", "args": {"x": 2}},
                },
                {"type": "respond", "message": "synthetic answer"},
                {"type": "reject"},
            ]
        },
    ):
        result = middleware.after_model(state, Runtime())

    assert result is not None
    # AI message + 2 synthetic ToolMessages (respond + reject; edit produces no message).
    assert len(result["messages"]) == 3

    updated_ai = result["messages"][0]
    assert isinstance(updated_ai, AIMessage)
    assert len(updated_ai.tool_calls) == 3
    edited_call = updated_ai.tool_calls[0]
    assert edited_call["args"] == {"x": 2}
    assert edited_call["id"] == "id_edit"

    respond_msg = result["messages"][1]
    assert isinstance(respond_msg, ToolMessage)
    assert respond_msg.tool_call_id == "id_resp"
    assert respond_msg.content == "synthetic answer"
    assert respond_msg.status == "success"

    reject_msg = result["messages"][2]
    assert isinstance(reject_msg, ToolMessage)
    assert reject_msg.tool_call_id == "id_rej"
    assert reject_msg.content == "declined by factory"
    assert reject_msg.status == "success"


def test_human_in_the_loop_middleware_rejection_response_callable_returns_non_toolmessage() -> None:
    """A factory returning something other than a `ToolMessage` raises `TypeError`."""

    def bad_factory(tool_call: ToolCall, decision: dict[str, Any]) -> Any:
        return "not a tool message"

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {
                "allowed_decisions": ["reject"],
                "rejection_response": bad_factory,
            }
        }
    )

    ai_message = AIMessage(
        content="I'll help",
        tool_calls=[{"name": "test_tool", "args": {}, "id": "tc-1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    with (
        patch(
            "langchain.agents.middleware.human_in_the_loop.interrupt",
            return_value={"decisions": [{"type": "reject"}]},
        ),
        pytest.raises(
            TypeError,
            match=re.escape(
                "`rejection_response` callable for tool 'test_tool' returned 'str'; "
                "expected a `ToolMessage`."
            ),
        ),
    ):
        middleware.after_model(state, Runtime())


def test_human_in_the_loop_middleware_rejection_response_callable_name_mismatch() -> None:
    """A factory returning a `ToolMessage` with the wrong `name` raises `ValueError`."""

    def bad_factory(tool_call: ToolCall, decision: dict[str, Any]) -> ToolMessage:
        return ToolMessage(
            content="declined",
            name="wrong_name",
            tool_call_id=tool_call["id"],
            status="success",
        )

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {
                "allowed_decisions": ["reject"],
                "rejection_response": bad_factory,
            }
        }
    )

    ai_message = AIMessage(
        content="I'll help",
        tool_calls=[{"name": "test_tool", "args": {}, "id": "tc-1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    with (
        patch(
            "langchain.agents.middleware.human_in_the_loop.interrupt",
            return_value={"decisions": [{"type": "reject"}]},
        ),
        pytest.raises(
            ValueError,
            match=re.escape(
                "`rejection_response` callable for tool 'test_tool' returned a "
                "`ToolMessage` with `name='wrong_name'`, but the rejected call's "
                "name is 'test_tool'."
            ),
        ),
    ):
        middleware.after_model(state, Runtime())


def test_human_in_the_loop_middleware_rejection_response_without_reject_raises() -> None:
    """Setting `rejection_response` without `'reject'` in `allowed_decisions` raises.

    Catches misconfiguration eagerly at construction rather than silently
    ignoring the override at runtime.
    """
    template = ToolMessage(
        content="should not be reachable",
        name="placeholder",
        tool_call_id="placeholder",
        status="error",
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Tool 'test_tool' has a `rejection_response` configured but 'reject' is "
            "not in `allowed_decisions`"
        ),
    ):
        HumanInTheLoopMiddleware(
            interrupt_on={
                "test_tool": {
                    "allowed_decisions": ["approve"],
                    "rejection_response": template,
                }
            }
        )
