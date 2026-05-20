import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.runtime import Runtime

from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.human_in_the_loop import (
    Action,
    HumanInTheLoopMiddleware,
)
from langchain.agents.middleware.types import AgentState, ToolCallRequest


def _make_request(
    tool_call: dict[str, Any],
    state: AgentState[Any] | None = None,
    runtime: Runtime | None = None,
) -> MagicMock:
    """Build a mock ToolCallRequest for wrap_tool_call tests."""
    req = MagicMock()
    req.tool_call = tool_call
    req.state = state or AgentState[Any](messages=[])
    req.runtime = runtime or Runtime()

    def _override(**kwargs: Any) -> MagicMock:
        new_tc = kwargs.get("tool_call", tool_call)
        return _make_request(new_tc, req.state, req.runtime)

    req.override.side_effect = _override
    return req


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


# ---------------------------------------------------------------------------
# interrupt_mode="per_call" — wrap_tool_call behaviour
# ---------------------------------------------------------------------------


def test_per_call_after_model_is_noop() -> None:
    """after_model returns None in per_call mode regardless of pending tool calls."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": True},
        interrupt_mode="per_call",
    )
    ai_message = AIMessage(
        content="...",
        tool_calls=[{"name": "test_tool", "args": {}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hi"), ai_message])
    result = middleware.after_model(state, Runtime())
    assert result is None


def test_batch_mode_wrap_tool_call_passthrough() -> None:
    """wrap_tool_call is a no-op passthrough in batch mode."""
    middleware = HumanInTheLoopMiddleware(interrupt_on={"test_tool": True})
    expected = ToolMessage(content="result", name="test_tool", tool_call_id="1")
    handler = MagicMock(return_value=expected)
    request = _make_request({"name": "test_tool", "args": {}, "id": "1"})

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt") as mock_interrupt:
        result = middleware.wrap_tool_call(request, handler)
        mock_interrupt.assert_not_called()

    handler.assert_called_once_with(request)
    assert result is expected


def test_per_call_approve() -> None:
    """Approve decision calls the handler unchanged."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve"]}},
        interrupt_mode="per_call",
    )
    expected = ToolMessage(content="ok", name="test_tool", tool_call_id="1")
    handler = MagicMock(return_value=expected)
    request = _make_request({"name": "test_tool", "args": {"x": 1}, "id": "1"})

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"type": "approve"},
    ):
        result = middleware.wrap_tool_call(request, handler)

    handler.assert_called_once()
    assert result is expected


def test_per_call_edit() -> None:
    """Edit decision calls handler with a modified request; original request is not mutated."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["edit"]}},
        interrupt_mode="per_call",
    )
    expected = ToolMessage(content="ok", name="test_tool", tool_call_id="1")
    handler = MagicMock(return_value=expected)
    request = _make_request({"name": "test_tool", "args": {"x": 1}, "id": "1"})

    edit_decision = {
        "type": "edit",
        "edited_action": Action(name="test_tool", args={"x": 99}),
    }
    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value=edit_decision,
    ):
        result = middleware.wrap_tool_call(request, handler)

    handler.assert_called_once()
    called_request = handler.call_args[0][0]
    assert called_request.tool_call["args"] == {"x": 99}
    assert called_request.tool_call["id"] == "1"
    assert result is expected


def test_per_call_reject() -> None:
    """Reject decision returns a synthetic error ToolMessage; handler is never called."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["reject"]}},
        interrupt_mode="per_call",
    )
    handler = MagicMock()
    request = _make_request({"name": "test_tool", "args": {}, "id": "1"})

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"type": "reject", "message": "Not allowed"},
    ):
        result = middleware.wrap_tool_call(request, handler)

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert result.content == "Not allowed"
    assert result.status == "error"
    assert result.tool_call_id == "1"


def test_per_call_respond() -> None:
    """Respond decision returns a synthetic success ToolMessage; handler is never called."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"ask_user": {"allowed_decisions": ["respond"]}},
        interrupt_mode="per_call",
    )
    handler = MagicMock()
    request = _make_request({"name": "ask_user", "args": {"q": "?"}, "id": "1"})

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"type": "respond", "message": "My answer"},
    ):
        result = middleware.wrap_tool_call(request, handler)

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert result.content == "My answer"
    assert result.status == "success"
    assert result.tool_call_id == "1"


def test_per_call_auto_approve_unknown_tool() -> None:
    """Tool not in interrupt_on passes straight through without an interrupt."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"known_tool": True},
        interrupt_mode="per_call",
    )
    expected = ToolMessage(content="r", name="unknown_tool", tool_call_id="1")
    handler = MagicMock(return_value=expected)
    request = _make_request({"name": "unknown_tool", "args": {}, "id": "1"})

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt") as mock_interrupt:
        result = middleware.wrap_tool_call(request, handler)
        mock_interrupt.assert_not_called()

    handler.assert_called_once_with(request)
    assert result is expected


def test_per_call_interrupt_payload_is_single_item() -> None:
    """Interrupt payload is a ToolCallReviewRequest (not a batched HITLRequest)."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve"]}},
        interrupt_mode="per_call",
    )
    handler = MagicMock(return_value=ToolMessage(content="ok", name="test_tool", tool_call_id="1"))
    request = _make_request({"name": "test_tool", "args": {"x": 1}, "id": "1"})

    captured: Any = None

    def capture(payload: Any) -> Any:
        nonlocal captured
        captured = payload
        return {"type": "approve"}

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=capture):
        middleware.wrap_tool_call(request, handler)

    assert captured is not None
    assert "action_request" in captured
    assert "review_config" in captured
    assert captured["action_request"]["name"] == "test_tool"
    assert captured["review_config"]["allowed_decisions"] == ["approve"]
    # Must NOT look like a batched HITLRequest
    assert "action_requests" not in captured
    assert "decisions" not in captured


# ---------------------------------------------------------------------------
# when predicate — batch and per_call modes
# ---------------------------------------------------------------------------


def test_when_predicate_batch_skips_interrupt_when_false() -> None:
    """`when` returning False prevents the tool call from joining the batch interrupt."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["approve"],
                when=lambda req: req.tool_call["args"].get("risky", False),
            )
        }
    )
    ai_message = AIMessage(
        content="...",
        tool_calls=[{"name": "test_tool", "args": {"risky": False}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hi"), ai_message])

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt") as mock_interrupt:
        result = middleware.after_model(state, Runtime())
        mock_interrupt.assert_not_called()

    assert result is None


def test_when_predicate_batch_fires_interrupt_when_true() -> None:
    """`when` returning True allows the tool call to trigger the batch interrupt."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["approve"],
                when=lambda req: req.tool_call["args"].get("risky", False),
            )
        }
    )
    ai_message = AIMessage(
        content="...",
        tool_calls=[{"name": "test_tool", "args": {"risky": True}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hi"), ai_message])

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "approve"}]},
    ):
        result = middleware.after_model(state, Runtime())

    assert result is not None


def test_when_predicate_per_call_skips_interrupt_when_false() -> None:
    """`when` returning False passes the call straight through in per_call mode."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["approve"],
                when=lambda req: req.tool_call["args"].get("risky", False),
            )
        },
        interrupt_mode="per_call",
    )
    expected = ToolMessage(content="ok", name="test_tool", tool_call_id="1")
    handler = MagicMock(return_value=expected)
    request = _make_request({"name": "test_tool", "args": {"risky": False}, "id": "1"})

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt") as mock_interrupt:
        result = middleware.wrap_tool_call(request, handler)
        mock_interrupt.assert_not_called()

    handler.assert_called_once_with(request)
    assert result is expected


def test_when_predicate_per_call_fires_interrupt_when_true() -> None:
    """`when` returning True triggers the interrupt in per_call mode."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["approve"],
                when=lambda req: req.tool_call["args"].get("risky", False),
            )
        },
        interrupt_mode="per_call",
    )
    expected = ToolMessage(content="ok", name="test_tool", tool_call_id="1")
    handler = MagicMock(return_value=expected)
    request = _make_request({"name": "test_tool", "args": {"risky": True}, "id": "1"})

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"type": "approve"},
    ):
        result = middleware.wrap_tool_call(request, handler)

    handler.assert_called_once()
    assert result is expected


def test_when_predicate_receives_correct_args() -> None:
    """The when predicate receives (tool_call, state, runtime) with correct values."""
    captured: list[Any] = []

    def capture_when(req: ToolCallRequest) -> bool:
        captured.append(req)
        return True

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["approve"],
                when=capture_when,
            )
        }
    )
    ai_message = AIMessage(
        content="...",
        tool_calls=[{"name": "test_tool", "args": {"val": 42}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hi"), ai_message])
    runtime = Runtime()

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "approve"}]},
    ):
        middleware.after_model(state, runtime)

    assert len(captured) == 1
    req = captured[0]
    assert req.tool_call["name"] == "test_tool"
    assert req.tool_call["args"] == {"val": 42}
    assert req.state is state
    assert req.runtime is runtime
