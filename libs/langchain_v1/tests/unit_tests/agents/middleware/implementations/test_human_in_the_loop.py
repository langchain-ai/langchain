import re
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from langchain.agents import create_agent
from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.human_in_the_loop import (
    Action,
    HumanInTheLoopMiddleware,
)
from langchain.agents.middleware.types import AgentState, ToolCallRequest
from tests.unit_tests.agents.model import FakeToolCallingModel


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


def test_human_in_the_loop_middleware_default_rejection_message() -> None:
    """Test reject decision default message discourages retries."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_response(_: Any) -> dict[str, Any]:
        return {"decisions": [{"type": "reject"}]}

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_response
    ):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert len(result["messages"]) == 2
        tool_message = result["messages"][1]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.content == (
            "User rejected the tool call for `test_tool` with id 1. "
            "The tool was not executed. Do not retry this tool call unless the user "
            "explicitly requests it."
        )
        assert tool_message.status == "error"
        assert tool_message.name == "test_tool"
        assert tool_message.tool_call_id == "1"


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

    def custom_description(tool_call: ToolCall, *_args: Any, **_kwargs: Any) -> str:
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
# when predicate
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

    with (
        patch("langchain.agents.middleware.human_in_the_loop.get_config", return_value={}),
        patch("langchain.agents.middleware.human_in_the_loop.interrupt") as mock_interrupt,
    ):
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

    with (
        patch("langchain.agents.middleware.human_in_the_loop.get_config", return_value={}),
        patch(
            "langchain.agents.middleware.human_in_the_loop.interrupt",
            return_value={"decisions": [{"type": "approve"}]},
        ),
    ):
        result = middleware.after_model(state, Runtime())

    assert result is not None


def test_when_predicate_receives_correct_args() -> None:
    """The when predicate receives a ToolCallRequest with correct values and a ToolRuntime."""
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
        tool_calls=[{"name": "test_tool", "args": {"val": 42}, "id": "tc-1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hi"), ai_message])
    runtime = Runtime()

    with (
        patch("langchain.agents.middleware.human_in_the_loop.get_config", return_value={}),
        patch(
            "langchain.agents.middleware.human_in_the_loop.interrupt",
            return_value={"decisions": [{"type": "approve"}]},
        ),
    ):
        middleware.after_model(state, runtime)

    assert len(captured) == 1
    req = captured[0]
    assert req.tool_call["name"] == "test_tool"
    assert req.tool_call["args"] == {"val": 42}
    assert req.tool is None
    assert req.state is state
    assert isinstance(req.runtime, ToolRuntime)
    assert req.runtime.tool_call_id == "tc-1"
    assert req.runtime.state is state
    assert req.runtime.context is runtime.context
    assert req.runtime.store is runtime.store


# ---------------------------------------------------------------------------
# interrupt_after (post-execution) tests
# ---------------------------------------------------------------------------

_INTERRUPT_PATH = "langchain.agents.middleware.human_in_the_loop.interrupt"


def _make_tool_request(
    tool_call: ToolCall,
    *,
    store: Any = None,
    state: AgentState[Any] | None = None,
) -> ToolCallRequest:
    """Build a `ToolCallRequest` with a `ToolRuntime` for `wrap_tool_call` tests."""
    request_state = state if state is not None else AgentState[Any](messages=[])
    runtime = ToolRuntime(
        state=request_state,
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id=tool_call["id"],
        store=store,
    )
    return ToolCallRequest(tool_call=tool_call, tool=None, state=request_state, runtime=runtime)


def test_interrupt_after_validation_rejects_before_decisions() -> None:
    """`interrupt_after` tools may only allow accept/replace decisions."""
    with pytest.raises(ValueError, match="may only contain"):
        HumanInTheLoopMiddleware(
            interrupt_on={
                "test_tool": InterruptOnConfig(allowed_decisions=["approve"], interrupt_after=True)
            }
        )


def test_before_interrupt_validation_rejects_after_decisions() -> None:
    """Before-execution tools may not use accept/replace decisions."""
    with pytest.raises(ValueError, match="require `interrupt_after=True`"):
        HumanInTheLoopMiddleware(
            interrupt_on={"test_tool": InterruptOnConfig(allowed_decisions=["accept"])}
        )


def test_after_model_skips_interrupt_after_tools() -> None:
    """`after_model` must not interrupt before an `interrupt_after` tool executes."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["accept", "replace"], interrupt_after=True
            )
        }
    )
    ai_message = AIMessage(
        content="...",
        tool_calls=[{"name": "test_tool", "args": {"x": 1}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hi"), ai_message])

    with patch(_INTERRUPT_PATH, side_effect=AssertionError("should not interrupt")):
        result = middleware.after_model(state, Runtime())

    assert result is None


def test_wrap_tool_call_after_accept_keeps_result() -> None:
    """Accepting keeps the executed tool result unchanged."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["accept", "replace"], interrupt_after=True
            )
        }
    )
    tool_call: ToolCall = {
        "name": "test_tool",
        "args": {"x": 1},
        "id": "1",
        "type": "tool_call",
    }
    request = _make_tool_request(tool_call)
    executed = ToolMessage(content="bus ack: job-42", name="test_tool", tool_call_id="1")

    def handler(_req: ToolCallRequest) -> ToolMessage:
        return executed

    with patch(_INTERRUPT_PATH, return_value={"decisions": [{"type": "accept"}]}):
        result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.content == "bus ack: job-42"


def test_wrap_tool_call_after_replace_substitutes_content() -> None:
    """Replacing returns a new `ToolMessage` with the provided content."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["accept", "replace"], interrupt_after=True
            )
        }
    )
    tool_call: ToolCall = {
        "name": "test_tool",
        "args": {"x": 1},
        "id": "1",
        "type": "tool_call",
    }
    request = _make_tool_request(tool_call)

    def handler(_req: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="placeholder", name="test_tool", tool_call_id="1")

    decision = {"decisions": [{"type": "replace", "message": "final result from worker"}]}
    with patch(_INTERRUPT_PATH, return_value=decision):
        result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.content == "final result from worker"
    assert result.status == "success"
    assert result.tool_call_id == "1"
    assert result.name == "test_tool"


def test_wrap_tool_call_passthrough_for_unconfigured_tool() -> None:
    """Tools without an `interrupt_after` config execute without interrupting."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "configured_tool": InterruptOnConfig(allowed_decisions=["accept"], interrupt_after=True)
        }
    )
    tool_call: ToolCall = {
        "name": "other_tool",
        "args": {},
        "id": "1",
        "type": "tool_call",
    }
    request = _make_tool_request(tool_call)
    sentinel = ToolMessage(content="ran", name="other_tool", tool_call_id="1")

    with patch(_INTERRUPT_PATH, side_effect=AssertionError("should not interrupt")):
        result = middleware.wrap_tool_call(request, lambda _req: sentinel)

    assert result is sentinel


def test_wrap_tool_call_passthrough_for_before_interrupt_tool() -> None:
    """Before-execution tools pass through `wrap_tool_call` (handled in `after_model`)."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": InterruptOnConfig(allowed_decisions=["approve"])}
    )
    tool_call: ToolCall = {
        "name": "test_tool",
        "args": {},
        "id": "1",
        "type": "tool_call",
    }
    request = _make_tool_request(tool_call)
    sentinel = ToolMessage(content="ran", name="test_tool", tool_call_id="1")

    with patch(_INTERRUPT_PATH, side_effect=AssertionError("should not interrupt")):
        result = middleware.wrap_tool_call(request, lambda _req: sentinel)

    assert result is sentinel


def test_wrap_tool_call_after_respects_when_predicate() -> None:
    """A falsy `when` predicate skips the post-execution interrupt."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["accept", "replace"],
                interrupt_after=True,
                when=lambda _req: False,
            )
        }
    )
    tool_call: ToolCall = {
        "name": "test_tool",
        "args": {},
        "id": "1",
        "type": "tool_call",
    }
    request = _make_tool_request(tool_call)
    sentinel = ToolMessage(content="ran", name="test_tool", tool_call_id="1")

    with (
        patch("langchain.agents.middleware.human_in_the_loop.get_config", return_value={}),
        patch(_INTERRUPT_PATH, side_effect=AssertionError("should not interrupt")),
    ):
        result = middleware.wrap_tool_call(request, lambda _req: sentinel)

    assert result is sentinel


def test_wrap_tool_call_after_rejects_disallowed_decision() -> None:
    """A decision not in `allowed_decisions` raises a `ValueError`."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(allowed_decisions=["accept"], interrupt_after=True)
        }
    )
    tool_call: ToolCall = {
        "name": "test_tool",
        "args": {},
        "id": "1",
        "type": "tool_call",
    }
    request = _make_tool_request(tool_call)

    def handler(_req: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="ran", name="test_tool", tool_call_id="1")

    decision = {"decisions": [{"type": "replace", "message": "nope"}]}
    with (
        patch(_INTERRUPT_PATH, return_value=decision),
        pytest.raises(ValueError, match="is not allowed for tool"),
    ):
        middleware.wrap_tool_call(request, handler)


def test_wrap_tool_call_after_caches_result_across_interrupt() -> None:
    """A configured store serves the tool result on resume without re-executing."""
    store = InMemoryStore()
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["accept", "replace"], interrupt_after=True
            )
        }
    )
    tool_call: ToolCall = {
        "name": "test_tool",
        "args": {},
        "id": "1",
        "type": "tool_call",
    }
    request = _make_tool_request(tool_call, store=store)
    calls = {"n": 0}

    def handler(_req: ToolCallRequest) -> ToolMessage:
        calls["n"] += 1
        return ToolMessage(content=f"run-{calls['n']}", name="test_tool", tool_call_id="1")

    class _HaltError(Exception):
        """Stand-in for the GraphInterrupt raised when the agent halts."""

    # First pass: the tool runs and the agent halts at the interrupt.
    with (
        patch(_INTERRUPT_PATH, side_effect=_HaltError()),
        pytest.raises(_HaltError),
    ):
        middleware.wrap_tool_call(request, handler)

    assert calls["n"] == 1
    assert store.get(middleware._cache_namespace, "1") is not None

    # Resume: the cached result is reused, so the tool is not invoked again.
    with patch(_INTERRUPT_PATH, return_value={"decisions": [{"type": "accept"}]}):
        result = middleware.wrap_tool_call(request, handler)

    assert calls["n"] == 1
    assert isinstance(result, ToolMessage)
    assert result.content == "run-1"
    assert store.get(middleware._cache_namespace, "1") is None


async def test_awrap_tool_call_after_replace_substitutes_content() -> None:
    """The async path executes the tool and applies the replace decision."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["accept", "replace"], interrupt_after=True
            )
        }
    )
    tool_call: ToolCall = {
        "name": "test_tool",
        "args": {},
        "id": "1",
        "type": "tool_call",
    }
    request = _make_tool_request(tool_call)

    async def handler(_req: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="placeholder", name="test_tool", tool_call_id="1")

    decision = {"decisions": [{"type": "replace", "message": "async final"}]}
    with patch(_INTERRUPT_PATH, return_value=decision):
        result = await middleware.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.content == "async final"


async def test_awrap_tool_call_after_caches_result_across_interrupt() -> None:
    """The async path reuses a cached result on resume without re-executing."""
    store = InMemoryStore()
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": InterruptOnConfig(
                allowed_decisions=["accept", "replace"], interrupt_after=True
            )
        }
    )
    tool_call: ToolCall = {
        "name": "test_tool",
        "args": {},
        "id": "1",
        "type": "tool_call",
    }
    request = _make_tool_request(tool_call, store=store)
    calls = {"n": 0}

    async def handler(_req: ToolCallRequest) -> ToolMessage:
        calls["n"] += 1
        return ToolMessage(content=f"run-{calls['n']}", name="test_tool", tool_call_id="1")

    class _HaltError(Exception):
        """Stand-in for the GraphInterrupt raised when the agent halts."""

    with (
        patch(_INTERRUPT_PATH, side_effect=_HaltError()),
        pytest.raises(_HaltError),
    ):
        await middleware.awrap_tool_call(request, handler)

    assert calls["n"] == 1

    with patch(_INTERRUPT_PATH, return_value={"decisions": [{"type": "accept"}]}):
        result = await middleware.awrap_tool_call(request, handler)

    assert calls["n"] == 1
    assert isinstance(result, ToolMessage)
    assert result.content == "run-1"


# ---------------------------------------------------------------------------
# End-to-end tests: real interrupt/resume through `create_agent`
# ---------------------------------------------------------------------------


def test_e2e_interrupt_after_replace_with_real_resume() -> None:
    """Drive a full interrupt/resume cycle through `create_agent`.

    Uses a real checkpointer + store so the interrupt and resume are handled by
    LangGraph (not a mock). Verifies the agent halts after the tool executes, the
    replaced content reaches the model, the tool is not re-invoked on resume, and the
    run completes.
    """
    calls = {"n": 0}

    @tool
    def submit_job(payload: str) -> str:
        """Submit a job to the event bus."""
        calls["n"] += 1
        return f"queued:{payload}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="submit_job", args={"payload": "abc"}, id="call-1")],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[submit_job],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "submit_job": InterruptOnConfig(
                        allowed_decisions=["accept", "replace"], interrupt_after=True
                    )
                }
            )
        ],
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )
    config = {"configurable": {"thread_id": "e2e-replace"}}

    interrupted = agent.invoke({"messages": [HumanMessage("Run the job")]}, config)
    assert "__interrupt__" in interrupted, "Expected an interrupt after the tool executed"
    assert calls["n"] == 1, "Tool should have executed exactly once before halting"

    final = agent.invoke(
        Command(resume={"decisions": [{"type": "replace", "message": "worker result"}]}),
        config,
    )

    assert "__interrupt__" not in final, "Agent should complete after resume"
    assert calls["n"] == 1, "Tool must not be re-invoked on resume (store-backed cache)"

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "worker result"
    assert tool_messages[0].status == "success"
    assert tool_messages[0].tool_call_id == "call-1"


def test_e2e_interrupt_after_accept_with_real_resume() -> None:
    """Accepting on resume keeps the executed tool result and completes the run."""
    calls = {"n": 0}

    @tool
    def submit_job(payload: str) -> str:
        """Submit a job to the event bus."""
        calls["n"] += 1
        return f"queued:{payload}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="submit_job", args={"payload": "abc"}, id="call-1")],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[submit_job],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "submit_job": InterruptOnConfig(
                        allowed_decisions=["accept", "replace"], interrupt_after=True
                    )
                }
            )
        ],
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )
    config = {"configurable": {"thread_id": "e2e-accept"}}

    interrupted = agent.invoke({"messages": [HumanMessage("Run the job")]}, config)
    assert "__interrupt__" in interrupted
    assert calls["n"] == 1

    final = agent.invoke(Command(resume={"decisions": [{"type": "accept"}]}), config)

    assert "__interrupt__" not in final
    assert calls["n"] == 1

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "queued:abc"


def test_e2e_interrupt_after_payload_structure() -> None:
    """The interrupt payload surfaces the executed result for review."""

    @tool
    def submit_job(payload: str) -> str:
        """Submit a job to the event bus."""
        return f"queued:{payload}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="submit_job", args={"payload": "abc"}, id="call-1")],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[submit_job],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "submit_job": InterruptOnConfig(
                        allowed_decisions=["accept", "replace"], interrupt_after=True
                    )
                }
            )
        ],
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )
    config = {"configurable": {"thread_id": "e2e-payload"}}

    interrupted = agent.invoke({"messages": [HumanMessage("Run the job")]}, config)

    interrupts = interrupted["__interrupt__"]
    assert len(interrupts) == 1
    hitl_request = interrupts[0].value
    action_requests = hitl_request["action_requests"]
    assert len(action_requests) == 1
    assert action_requests[0]["name"] == "submit_job"
    assert action_requests[0]["args"] == {"payload": "abc"}
    assert "queued:abc" in action_requests[0]["description"]
    assert hitl_request["review_configs"][0]["allowed_decisions"] == ["accept", "replace"]


def test_e2e_interrupt_after_without_store_still_works() -> None:
    """Without a store the flow still completes (idempotency is the only trade-off)."""
    calls = {"n": 0}

    @tool
    def submit_job(payload: str) -> str:
        """Submit a job to the event bus."""
        calls["n"] += 1
        return f"queued:{payload}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="submit_job", args={"payload": "abc"}, id="call-1")],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[submit_job],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "submit_job": InterruptOnConfig(
                        allowed_decisions=["accept", "replace"], interrupt_after=True
                    )
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "e2e-no-store"}}

    interrupted = agent.invoke({"messages": [HumanMessage("Run the job")]}, config)
    assert "__interrupt__" in interrupted

    final = agent.invoke(
        Command(resume={"decisions": [{"type": "replace", "message": "worker result"}]}),
        config,
    )
    assert "__interrupt__" not in final
    # Without a store the tool re-executes on resume (documented trade-off): the
    # node re-runs from the start before the cached human decision is applied.
    assert calls["n"] == 2
    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "worker result"
