import re
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated, Any, cast
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode, ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import AfterValidator, TypeAdapter, ValidationError

from langchain.agents.factory import _make_tools_to_model_edge, create_agent
from langchain.agents.middleware import InterruptOnConfig, ToolErrorMiddleware, ToolRetryMiddleware
from langchain.agents.middleware.human_in_the_loop import (
    _EDIT_NOTICE,
    _EDITED_TOOL_CALLS_KEY,
    Action,
    Decision,
    DecisionType,
    HumanInTheLoopMiddleware,
    InterruptMode,
    _decision_schema,
    _HumanInTheLoopState,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    InputAgentState,
    OutputAgentState,
    ToolCallRequest,
)
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

_EXPECTED_NOTICE = (
    f'{_EDIT_NOTICE} Executed instead: write_file_tool with arguments {{"content": "edited"}}.'
)
_EXPECTED_NOTICE_WITH_CONTENT = f"{_EXPECTED_NOTICE}\n\nTool response:"


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


def test_human_in_the_loop_middleware_rejects_empty_allowed_decisions() -> None:
    """Test that an empty `allowed_decisions` list raises instead of silently disabling the gate."""
    with pytest.raises(ValueError, match="test_tool"):
        HumanInTheLoopMiddleware(interrupt_on={"test_tool": {"allowed_decisions": []}})


def test_human_in_the_loop_middleware_rejects_missing_allowed_decisions() -> None:
    """Test that a config missing `allowed_decisions` (e.g. `when`-only) raises."""
    with pytest.raises(ValueError, match="test_tool"):
        HumanInTheLoopMiddleware(
            interrupt_on={"test_tool": {"when": lambda _req: True}}  # type: ignore[dict-item]
        )


def test_human_in_the_loop_middleware_rejects_typoed_key() -> None:
    """Test that a misspelled `allowed_decisions` key raises instead of being silently dropped."""
    with pytest.raises(ValueError, match="test_tool"):
        HumanInTheLoopMiddleware(
            interrupt_on={"test_tool": {"alowed_decisions": ["approve"]}}  # type: ignore[dict-item]
        )


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
        assert result["messages"][0].tool_calls[0]["args"] == {"input": "test"}
        assert result["messages"][0].tool_calls[0]["id"] == "1"  # ID should be preserved
        assert result[_EDITED_TOOL_CALLS_KEY] == {
            "1": {"name": "test_tool", "args": {"input": "edited"}}
        }


def test_human_in_the_loop_middleware_single_tool_rejection_reason() -> None:
    """Test a custom rejection reason retains its human-provided context."""
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
        assert result["messages"][1].content == (
            "User rejected the tool call for `test_tool` with reason: Custom response message"
        )
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


def _assert_tool_messages_are_paired(messages: list[Any]) -> None:
    """Assert every `ToolMessage` answers a call declared by the preceding `AIMessage`.

    Model providers (e.g. OpenAI) reject a request where a `ToolMessage`'s
    `tool_call_id` has no matching entry in the immediately preceding `AIMessage`'s
    `tool_calls`, that pairing must hold for every model turn, not just the last.
    """
    pending_ids: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            pending_ids = {tc["id"] for tc in message.tool_calls if tc["id"] is not None}
        elif isinstance(message, ToolMessage):
            assert message.tool_call_id in pending_ids, (
                f"ToolMessage {message.tool_call_id!r} has no matching tool call in the "
                "preceding AIMessage"
            )
            pending_ids.discard(message.tool_call_id)


def test_human_in_the_loop_middleware_rejected_call_not_executed_and_stays_paired() -> None:
    """A rejected tool call must never execute, and the message history must stay valid.

    Exercises the real `create_agent` graph (not just the middleware in isolation) to
    confirm two things at once: the underlying tool is never invoked, and the agent can
    still take its next model turn afterward which requires every `ToolMessage` to
    answer a tool call the preceding `AIMessage` actually declared.
    """
    calls: list[str] = []

    @tool
    def risky_tool(value: str) -> str:
        """A tool that would be dangerous to run without approval."""
        calls.append(value)
        return f"Executed: {value}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="risky_tool", args={"value": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[risky_tool],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"risky_tool": {"allowed_decisions": ["approve", "reject"]}}
            )
        ],
        checkpointer=InMemorySaver(),
    )
    interrupted = agent.invoke(
        {"messages": [HumanMessage("Please run risky_tool")]},
        {"configurable": {"thread_id": "reject-not-executed"}},
    )
    assert "__interrupt__" in interrupted

    final = agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "denied"}]}),
        {"configurable": {"thread_id": "reject-not-executed"}},
    )

    # The graph must complete the next model turn rather than getting stuck or erroring.
    assert "__interrupt__" not in final
    # The tool itself must never run.
    assert calls == []
    # The message history must remain protocol-valid throughout, including the rejection turn.
    _assert_tool_messages_are_paired(final["messages"])


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
        # Keep rejected calls in the `AIMessage` for protocol validity: their
        # `ToolMessage` must reference a preceding tool call. The graph still prevents
        # the rejected call from executing.
        updated_ai_message = result["messages"][0]
        assert len(updated_ai_message.tool_calls) == 2
        assert updated_ai_message.tool_calls[0]["name"] == "get_forecast"  # Accepted
        assert updated_ai_message.tool_calls[1]["name"] == "get_temperature"  # Rejected, kept

        # Second message should be the tool message for the rejected tool call
        tool_message = result["messages"][1]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.content == (
            "User rejected the tool call for `get_temperature` with reason: "
            "User rejected this tool call"
        )
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
        assert updated_ai_message.tool_calls[0]["args"] == {"location": "San Francisco"}
        assert updated_ai_message.tool_calls[0]["id"] == "1"  # ID preserved
        assert updated_ai_message.tool_calls[1]["args"] == {"location": "San Francisco"}
        assert updated_ai_message.tool_calls[1]["id"] == "2"  # ID preserved
        assert result[_EDITED_TOOL_CALLS_KEY]["1"]["args"] == {"location": "New York"}


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

        # The model's own call is preserved; the reviewer's is recorded for execution.
        updated_ai_message = result["messages"][0]
        assert updated_ai_message.tool_calls[0]["args"] == {"input": "test"}
        assert updated_ai_message.tool_calls[0]["id"] == "1"  # ID preserved
        assert result[_EDITED_TOOL_CALLS_KEY] == {
            "1": {"name": "test_tool", "args": {"input": "modified"}}
        }


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
        assert result["messages"][0].tool_calls[0]["args"] == {"input": "test"}
        assert result[_EDITED_TOOL_CALLS_KEY]["1"]["args"] == {"input": "edited"}

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
        assert updated_ai_message.tool_calls[1]["args"] == {"val": 2}  # model's own
        assert result[_EDITED_TOOL_CALLS_KEY]["id_b"]["args"] == {"val": 200}  # reviewer's
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
        # tool_b is still declared on the AIMessage (rejection is handled via the paired
        # ToolMessage, not by stripping the call -- see
        # test_human_in_the_loop_middleware_rejected_call_not_executed_and_stays_paired).
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
        assert tool_message.content == (
            "User rejected the tool call for `tool_b` with reason: Rejected tool B"
        )
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

    # Called directly, outside a graph: there's no run config, and `when` still runs.
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
    assert req.runtime.tools == []
    assert req.runtime.state is state
    assert req.runtime.context is runtime.context
    assert req.runtime.store is runtime.store


def test_human_in_the_loop_middleware_edit_annotates_tool_result() -> None:
    """An edited call runs the reviewer's args and its result is attributed to them."""
    executed: list[dict[str, Any]] = []

    @tool
    def write_file_tool(path: str, content: str) -> str:
        """Write content to a file."""
        executed.append({"path": path, "content": content})
        return f"File written to {path}"

    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(
                    name="write_file_tool",
                    args={"path": "notes.txt", "content": "Hello, world!"},
                    id="1",
                )
            ],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[write_file_tool],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"write_file_tool": {"allowed_decisions": ["approve", "edit"]}}
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "edit-annotates-result"}}

    interrupted = agent.invoke(
        {"messages": [HumanMessage("Write notes.txt with 'Hello, world!'")]}, config
    )
    assert "__interrupt__" in interrupted

    final = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "write_file_tool",
                            "args": {"path": "notes.txt", "content": "reviewer value"},
                        },
                    }
                ]
            }
        ),
        config,
    )

    assert executed == [{"path": "notes.txt", "content": "reviewer value"}]

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    content = tool_messages[0].content
    assert isinstance(content, str)
    assert content.endswith("File written to notes.txt")
    assert _EDIT_NOTICE in content
    # The original, untrusted args must not be echoed back.
    assert "Hello, world!" not in content
    assert "__interrupt__" not in final
    _assert_tool_messages_are_paired(final["messages"])


def test_human_in_the_loop_middleware_approve_does_not_annotate() -> None:
    """An approved call was the model's own, so its result must not be annotated."""

    @tool
    def write_file_tool(path: str, content: str) -> str:
        """Write content to a file."""
        return f"File written to {path} ({len(content)} chars)"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="write_file_tool", args={"path": "/p", "content": "c"}, id="1")],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[write_file_tool],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"write_file_tool": {"allowed_decisions": ["approve", "edit"]}}
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "approve-no-annotation"}}
    agent.invoke({"messages": [HumanMessage("write it")]}, config)
    final = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].content == "File written to /p (1 chars)"


def _edited_request(tool_call_id: str = "1") -> ToolCallRequest:
    """A `ToolCallRequest` whose call a reviewer edited."""
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "write_file_tool", "args": {"content": "edited"}, "id": tool_call_id}],
    )
    return ToolCallRequest(
        tool_call=ToolCall(name="write_file_tool", args={"content": "edited"}, id=tool_call_id),
        tool=None,
        state=_HumanInTheLoopState[Any](
            messages=[HumanMessage("go"), ai_message],
            hitl_edited_tool_calls={
                tool_call_id: Action(name="write_file_tool", args={"content": "edited"})
            },
        ),
        runtime=None,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("tool_output", "expected_notice_block"),
    [
        (
            [{"type": "text", "text": "wrote it"}],
            {"type": "text", "text": _EXPECTED_NOTICE_WITH_CONTENT},
        ),
        (
            [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "u"}}],
            {"type": "text", "text": _EXPECTED_NOTICE_WITH_CONTENT},
        ),
        (["wrote it"], _EXPECTED_NOTICE_WITH_CONTENT),
        ([], {"type": "text", "text": _EXPECTED_NOTICE}),  # no label without content
    ],
    ids=["text-block", "mixed-blocks", "plain-strings", "empty"],
)
def test_human_in_the_loop_middleware_edit_annotates_list_content(
    tool_output: list[Any], expected_notice_block: Any
) -> None:
    """Block-content results are annotated, preserving existing blocks and their shape."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"write_file_tool": {"allowed_decisions": ["edit"]}}
    )
    result = ToolMessage(content=tool_output, tool_call_id="1", name="write_file_tool")

    annotated = middleware.wrap_tool_call(_edited_request(), lambda _: result)

    assert isinstance(annotated, ToolMessage)
    assert annotated.content == [expected_notice_block, *tool_output]


def test_human_in_the_loop_middleware_edit_annotates_command_result() -> None:
    """A `Command` result has its own `ToolMessage` annotated, leaving others intact."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"write_file_tool": {"allowed_decisions": ["edit"]}}
    )
    unrelated = ToolMessage(content="other", tool_call_id="99", name="other_tool")
    command: Command[Any] = Command(
        update={
            "messages": [
                ToolMessage(content="wrote it", tool_call_id="1", name="write_file_tool"),
                unrelated,
            ],
            "some_state_key": "preserved",
        }
    )

    result = middleware.wrap_tool_call(_edited_request(), lambda _: command)

    assert isinstance(result, Command)
    assert isinstance(result.update, dict)
    assert result.update["some_state_key"] == "preserved"
    annotated, passthrough = result.update["messages"]
    assert annotated.content == f"{_EXPECTED_NOTICE_WITH_CONTENT}\nwrote it"
    assert passthrough.content == "other"


def test_human_in_the_loop_middleware_edit_notice_is_not_duplicated() -> None:
    """The notice is idempotent; retry middleware may re-invoke the handler."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"write_file_tool": {"allowed_decisions": ["edit"]}}
    )
    request = _edited_request()
    once = middleware.wrap_tool_call(
        request, lambda _: ToolMessage(content="wrote it", tool_call_id="1")
    )
    assert isinstance(once, ToolMessage)
    twice = middleware.wrap_tool_call(request, lambda _: once)

    assert isinstance(twice, ToolMessage)
    assert twice.content == once.content
    assert twice.content.count(_EDIT_NOTICE) == 1


async def test_human_in_the_loop_middleware_edit_annotates_async() -> None:
    """`awrap_tool_call` must behave identically to the sync hook."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"write_file_tool": {"allowed_decisions": ["edit"]}}
    )

    async def handler(_: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="wrote it", tool_call_id="1", name="write_file_tool")

    result = await middleware.awrap_tool_call(_edited_request(), handler)

    assert isinstance(result, ToolMessage)
    assert result.content == f"{_EXPECTED_NOTICE_WITH_CONTENT}\nwrote it"


def test_human_in_the_loop_middleware_edit_notice_is_customizable() -> None:
    """`edit_notice` replaces the default text."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"write_file_tool": {"allowed_decisions": ["edit"]}},
        edit_notice="Operator overrode these args.",
    )

    result = middleware.wrap_tool_call(
        _edited_request(), lambda _: ToolMessage(content="wrote it", tool_call_id="1")
    )

    assert isinstance(result, ToolMessage)
    assert isinstance(result.content, str)
    assert result.content.startswith("Operator overrode these args.")
    assert _EDIT_NOTICE not in result.content


@pytest.mark.parametrize(
    "tool_output",
    ["wrote it", [{"type": "text", "text": "wrote it"}]],
    ids=["string", "blocks"],
)
def test_human_in_the_loop_middleware_edit_notice_can_be_disabled(tool_output: Any) -> None:
    """`edit_notice=None` leaves results untouched for both content shapes."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"write_file_tool": {"allowed_decisions": ["edit"]}},
        edit_notice=None,
    )

    result = middleware.wrap_tool_call(
        _edited_request(), lambda _: ToolMessage(content=tool_output, tool_call_id="1")
    )

    assert isinstance(result, ToolMessage)
    assert result.content == tool_output


def test_human_in_the_loop_middleware_edit_executes_reviewers_call() -> None:
    """The reviewer's args run while the model's own call stays in the message."""
    executed: list[dict[str, Any]] = []

    @tool
    def write_file_tool(path: str, content: str) -> str:
        """Write content to a file."""
        executed.append({"path": path, "content": content})
        return f"File written to {path}"

    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(
                    name="write_file_tool", args={"path": "notes.txt", "content": "mine"}, id="1"
                )
            ],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[write_file_tool],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"write_file_tool": {"allowed_decisions": ["approve", "edit"]}}
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "executes-reviewer-call"}}
    agent.invoke({"messages": [HumanMessage("write it")]}, config)
    final = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "write_file_tool",
                            "args": {"path": "notes.txt", "content": "reviewers"},
                        },
                    }
                ]
            }
        ),
        config,
    )

    assert executed == [{"path": "notes.txt", "content": "reviewers"}]
    ai_message = next(m for m in final["messages"] if isinstance(m, AIMessage) and m.tool_calls)
    assert ai_message.tool_calls[0]["args"] == {"path": "notes.txt", "content": "mine"}
    tool_message = next(m for m in final["messages"] if isinstance(m, ToolMessage))
    assert "reviewers" in tool_message.content
    _assert_tool_messages_are_paired(final["messages"])


def test_human_in_the_loop_middleware_edit_can_redirect_to_another_tool() -> None:
    """A reviewer may redirect the call to a different tool."""
    executed: list[str] = []

    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        executed.append("send_email")
        return f"sent to {to}"

    @tool
    def draft_email(to: str) -> str:
        """Draft an email."""
        executed.append("draft_email")
        return f"drafted to {to}"

    model = FakeToolCallingModel(
        tool_calls=[[ToolCall(name="send_email", args={"to": "a@b.c"}, id="1")], []]
    )
    agent = create_agent(
        model=model,
        tools=[send_email, draft_email],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"send_email": {"allowed_decisions": ["approve", "edit"]}}
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "edit-redirects-tool"}}
    agent.invoke({"messages": [HumanMessage("send it")]}, config)
    final = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {"name": "draft_email", "args": {"to": "a@b.c"}},
                    }
                ]
            }
        ),
        config,
    )

    assert executed == ["draft_email"]
    ai_message = next(m for m in final["messages"] if isinstance(m, AIMessage) and m.tool_calls)
    assert ai_message.tool_calls[0]["name"] == "send_email"
    tool_message = next(m for m in final["messages"] if isinstance(m, ToolMessage))
    assert "drafted" in tool_message.content
    assert "draft_email" in tool_message.content  # the notice names what ran


@pytest.mark.parametrize(
    ("requested", "replacement", "expected_turns_after_tool"),
    [("direct_tool", "normal_tool", 1), ("normal_tool", "direct_tool", 0)],
    ids=["direct-to-normal", "normal-to-direct"],
)
def test_human_in_the_loop_middleware_edit_routes_on_executed_tool(
    requested: str, replacement: str, expected_turns_after_tool: int
) -> None:
    """`return_direct` termination must follow the tool that ran, not the one requested."""

    @tool(return_direct=True)
    def direct_tool(x: str) -> str:
        """Return directly."""
        return f"direct {x}"

    @tool
    def normal_tool(x: str) -> str:
        """Do not return directly."""
        return f"normal {x}"

    model = FakeToolCallingModel(
        tool_calls=[[ToolCall(name=requested, args={"x": "1"}, id="1")], []]
    )
    agent = create_agent(
        model=model,
        tools=[direct_tool, normal_tool],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={requested: {"allowed_decisions": ["approve", "edit"]}}
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": f"route-{requested}-{replacement}"}}
    agent.invoke({"messages": [HumanMessage("go")]}, config)
    final = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "edit", "edited_action": {"name": replacement, "args": {"x": "1"}}}
                ]
            }
        ),
        config,
    )

    tool_idx = max(i for i, m in enumerate(final["messages"]) if isinstance(m, ToolMessage))
    model_turns = sum(isinstance(m, AIMessage) for m in final["messages"][tool_idx + 1 :])
    assert model_turns == expected_turns_after_tool


def test_human_in_the_loop_middleware_edit_to_unknown_tool_raises() -> None:
    """A reviewer naming a tool the agent does not have fails loudly."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"write_file_tool": {"allowed_decisions": ["edit"]}}
    )

    @tool
    def write_file_tool(content: str) -> str:
        """Write content."""
        return f"wrote {len(content)} chars"

    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "write_file_tool", "args": {"content": "x"}, "id": "1"}],
    )
    request = ToolCallRequest(
        tool_call=ToolCall(name="write_file_tool", args={"content": "x"}, id="1"),
        tool=write_file_tool,
        state=_HumanInTheLoopState[Any](
            messages=[HumanMessage("go"), ai_message],
            hitl_edited_tool_calls={"1": Action(name="nope", args={"content": "y"})},
        ),
        runtime=SimpleNamespace(tools=[write_file_tool]),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="not an available tool"):
        middleware.wrap_tool_call(
            request, lambda _: ToolMessage(content="wrote it", tool_call_id="1")
        )


def test_a_turn_that_resolves_no_decisions_clears_the_previous_turns_edits() -> None:
    """A recorded edit does not outlive the turn that recorded it.

    State is not scoped to a message the way `response_metadata` was, and providers are
    not required to keep tool call IDs unique across turns. `after_model` therefore
    drops the previous turn's edits the moment it reaches a turn with nothing to review,
    so a reused ID cannot pull an earlier edit onto a call that never went to a human.
    """
    middleware = HumanInTheLoopMiddleware(interrupt_on={"write_file_tool": True})
    state = _HumanInTheLoopState[Any](
        messages=[
            HumanMessage("go"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file_tool", "args": {"path": "p"}, "id": "1"}],
            ),
        ],
        hitl_edited_tool_calls={"1": Action(name="write_file_tool", args={"content": "reviewer"})},
    )

    assert middleware.after_model(state, Runtime()) == {_EDITED_TOOL_CALLS_KEY: {}}


def test_a_turn_with_nothing_recorded_does_not_write_state() -> None:
    """The clear is only issued when there is something to clear."""
    middleware = HumanInTheLoopMiddleware(interrupt_on={"write_file_tool": True})
    state = _HumanInTheLoopState[Any](
        messages=[
            HumanMessage("go"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file_tool", "args": {"path": "p"}, "id": "1"}],
            ),
        ],
    )

    assert middleware.after_model(state, Runtime()) is None


def test_reused_tool_call_id_does_not_replay_a_previous_turns_edit() -> None:
    """End to end: a later call reusing an edited call's ID runs the model's own args."""
    executed: list[tuple[str, str]] = []

    @tool
    def write_file_tool(path: str, content: str) -> str:
        """Write content to a file."""
        executed.append(("write", content))
        return f"File written to {path}"

    @tool
    def read_file_tool(path: str) -> str:
        """Read a file."""
        executed.append(("read", path))
        return f"Contents of {path}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="write_file_tool", args={"path": "a.txt", "content": "model"}, id="1")],
            # The same ID on a tool that is not gated, so this turn reviews nothing.
            [ToolCall(name="read_file_tool", args={"path": "b.txt"}, id="1")],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[write_file_tool, read_file_tool],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file_tool": True})],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "reused-tool-call-id"}}

    agent.invoke({"messages": [HumanMessage("write it, then read it")]}, config)
    final = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "write_file_tool",
                            "args": {"path": "a.txt", "content": "reviewer"},
                        },
                    }
                ]
            }
        ),
        config,
    )

    assert executed == [("write", "reviewer"), ("read", "b.txt")]
    read_result = [m for m in final["messages"] if isinstance(m, ToolMessage)][-1]
    # The read was never edited, so its result carries no notice.
    assert read_result.content == "Contents of b.txt"


def test_recorded_edit_survives_a_later_tightening_of_allowed_decisions() -> None:
    """A completed decision is honored as recorded, even under a stricter config.

    `allowed_decisions` governs what a reviewer may decide at review time. Re-checking
    it at execution time would not deny the edit: the only thing left to run would be
    the model's original call, which is precisely what the reviewer declined.
    """
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"send_email_tool": {"allowed_decisions": ["approve", "reject"]}}
    )

    @tool
    def send_email_tool(to: str) -> str:
        """Send an email."""
        return f"sent to {to}"

    @tool
    def draft_email_tool(to: str) -> str:
        """Draft an email without sending it."""
        return f"drafted to {to}"

    # Recorded while `edit` was still permitted; the config has since been tightened.
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "send_email_tool", "args": {"to": "a@b.c"}, "id": "1"}],
    )
    request = ToolCallRequest(
        tool_call=ToolCall(name="send_email_tool", args={"to": "a@b.c"}, id="1"),
        tool=send_email_tool,
        state=_HumanInTheLoopState[Any](
            messages=[HumanMessage("go"), ai_message],
            hitl_edited_tool_calls={"1": Action(name="draft_email_tool", args={"to": "a@b.c"})},
        ),
        runtime=SimpleNamespace(tools=[send_email_tool, draft_email_tool]),  # type: ignore[arg-type]
    )

    executed: list[ToolCallRequest] = []

    def handler(req: ToolCallRequest) -> ToolMessage:
        executed.append(req)
        assert req.tool is not None
        return ToolMessage(content=req.tool.invoke(req.tool_call["args"]), tool_call_id="1")

    result = middleware.wrap_tool_call(request, handler)

    # The reviewer's replacement runs; the declined original never does.
    assert [req.tool_call["name"] for req in executed] == ["draft_email_tool"]
    assert executed[0].tool is draft_email_tool
    assert isinstance(result, ToolMessage)
    assert "drafted to a@b.c" in result.content
    assert "draft_email_tool" in result.content  # the notice names what ran


async def test_async_turn_that_resolves_no_decisions_clears_the_previous_turns_edits() -> None:
    """`aafter_model` drops the previous turn's edits too."""
    middleware = HumanInTheLoopMiddleware(interrupt_on={"write_file_tool": True})
    state = _HumanInTheLoopState[Any](
        messages=[
            HumanMessage("go"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file_tool", "args": {"path": "p"}, "id": "1"}],
            ),
        ],
        hitl_edited_tool_calls={"1": Action(name="write_file_tool", args={"content": "reviewer"})},
    )

    assert await middleware.aafter_model(state, Runtime()) == {_EDITED_TOOL_CALLS_KEY: {}}


def test_return_direct_routing_keeps_calls_with_unnamed_results() -> None:
    """A result without a usable name must still participate in the return-direct check."""

    @tool(return_direct=True)
    def direct_tool(x: str) -> str:
        """Return directly."""
        return f"direct {x}"

    @tool
    def normal_tool(x: str) -> str:
        """Do not return directly."""
        return f"normal {x}"

    node = ToolNode([direct_tool, normal_tool])
    edge = _make_tools_to_model_edge(
        tool_node=node,
        model_destination="MODEL",
        structured_output_tools={},
        end_destination="END",
    )
    ai_message = AIMessage(
        "",
        tool_calls=[
            {"name": "direct_tool", "args": {"x": "1"}, "id": "1", "type": "tool_call"},
            {"name": "normal_tool", "args": {"x": "2"}, "id": "2", "type": "tool_call"},
        ],
    )
    messages = [
        HumanMessage("go"),
        ai_message,
        ToolMessage(content="direct result", tool_call_id="1", name="direct_tool"),
        # A tool or middleware may omit `name`; the call must not drop out of the check.
        ToolMessage(content="normal result", tool_call_id="2"),
    ]

    assert edge({"messages": messages}) == "MODEL"


# --- Per-call mode (interrupt_mode="per_call") ---


@pytest.mark.parametrize("allowed", [["respond"], ["respond", "respond"]])
def test_decision_schema_with_one_decision_is_a_plain_object(allowed: list[DecisionType]) -> None:
    schema = TypeAdapter(_decision_schema(allowed, "send_email")).json_schema()
    assert schema["properties"]["type"]["const"] == "respond"
    assert schema["required"] == ["type", "message"]


@pytest.mark.parametrize(
    ("answer", "loc", "error_type"),
    [
        ({"type": "edit"}, ("edit", "edited_action"), "missing"),
        ({"type": "nope"}, (), "union_tag_invalid"),
        (
            {"type": "edit", "edited_action": {"name": "send_email", "args": {}}},
            ("edit", "edited_action", "args", "to"),
            "missing",
        ),
        (
            {"type": "edit", "edited_action": {"name": "delete_file", "args": {"to": "b"}}},
            ("edit", "edited_action", "name"),
            "literal_error",
        ),
        (
            {
                "type": "edit",
                "edited_action": {"name": "send_email", "args": {"to": "b", "ccc": 1}},
            },
            ("edit", "edited_action", "args", "ccc"),
            "extra_forbidden",
        ),
        (
            {
                "type": "edit",
                "edited_action": {"name": "send_email", "args": {"to": "b"}, "nmae": 1},
            },
            ("edit", "edited_action", "nmae"),
            "extra_forbidden",
        ),
    ],
)
def test_decision_schema_rejects_a_bad_answer_with_one_error_at_the_problem(
    answer: object, loc: tuple[str, ...], error_type: str
) -> None:
    @tool
    def send_email(to: str, cc: str | None = None) -> str:
        """Send an email."""
        return f"sent to {to}, cc {cc}"

    schema = _decision_schema(["approve", "edit", "reject", "respond"], "send_email", send_email)
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(schema).validate_python(answer)
    assert [(e["loc"], e["type"]) for e in exc_info.value.errors()] == [(loc, error_type)]


@pytest.mark.parametrize(
    "tool",
    [
        None,
        StructuredTool(
            name="send_email",
            description="d",
            func=lambda **_: "sent",
            args_schema={"type": "object", "properties": {"to": {"type": "string"}}},
        ),
    ],
    ids=["no_tool", "json_schema_tool"],
)
def test_edit_args_without_a_pydantic_schema_are_shown_but_not_enforced(
    tool: BaseTool | None,
) -> None:
    adapter = TypeAdapter(_decision_schema(["edit"], "send_email", tool))
    assert adapter.json_schema()["$defs"]["EditedAction"]["properties"]["args"]["type"] == "object"
    answer = {"type": "edit", "edited_action": {"name": "send_email", "args": {"to": 5}}}
    assert adapter.validate_python(answer) == answer


def test_middleware_checks_interrupt_mode() -> None:
    assert HumanInTheLoopMiddleware(interrupt_on={"t": True}).interrupt_mode == "batched"
    with pytest.raises(ValueError, match="must be 'batched' or 'per_call', got 'sometimes'"):
        HumanInTheLoopMiddleware(
            interrupt_on={"t": True}, interrupt_mode=cast("InterruptMode", "sometimes")
        )


def _agent(
    tools: list[BaseTool],
    tool_calls: list[ToolCall],
    interrupt_on: dict[str, bool | InterruptOnConfig],
    *after: AgentMiddleware,
) -> CompiledStateGraph[AgentState[Any], None, InputAgentState, OutputAgentState[Any]]:
    """An agent with per-call HITL whose model makes `tool_calls`, then finishes."""
    return create_agent(
        model=FakeToolCallingModel(tool_calls=[tool_calls, []]),
        tools=tools,
        middleware=[HumanInTheLoopMiddleware(interrupt_on, interrupt_mode="per_call"), *after],
        checkpointer=InMemorySaver(),
    )


def test_per_call_interrupt_shows_the_call_and_the_answers_it_accepts() -> None:
    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        return f"sent to {to}"

    agent = _agent(
        [send_email],
        [ToolCall(name="send_email", args={"to": "alice"}, id="call_email")],
        {
            "send_email": {
                "allowed_decisions": ["approve", "edit", "reject", "respond"],
                "description": "Email",
                # Not used in per-call mode: the edit's args come from the tool itself.
                "args_schema": {"type": "object", "properties": {"recipient": {"type": "string"}}},
            }
        },
    )
    result = agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t"}})
    [intr] = result["__interrupt__"]

    # `value`: the tool call waiting for review.
    assert intr.value == {
        "type": "tool_approval",
        "tool_call_id": "call_email",
        "name": "send_email",
        "args": {"to": "alice"},
        "description": "Email",
    }
    # `response_schema`: one branch per allowed decision, matched on `type`.
    schema = intr.response_schema
    defs = schema["$defs"]
    assert schema["oneOf"] == [
        {"$ref": "#/$defs/ApproveDecision"},
        {"$ref": "#/$defs/EditDecision"},
        {"$ref": "#/$defs/RejectDecision"},
        {"$ref": "#/$defs/RespondDecision"},
    ]
    assert defs["ApproveDecision"]["required"] == ["type"]
    assert defs["EditDecision"]["required"] == ["type", "edited_action"]
    assert defs["RejectDecision"]["required"] == ["type"]  # `message` is optional
    assert defs["RespondDecision"]["required"] == ["type", "message"]
    # An edit names the same tool, and its args follow the tool's own schema.
    edit, edited_action, args = defs["EditDecision"], defs["EditedAction"], defs["send_email"]
    assert edit["properties"]["edited_action"] == {"$ref": "#/$defs/EditedAction"}
    assert edited_action["properties"]["name"]["const"] == "send_email"
    assert edited_action["properties"]["args"] == {"$ref": "#/$defs/send_email"}
    assert (list(args["properties"]), args["required"]) == (["to"], ["to"])
    assert args["description"] == "Send an email."
    # Unknown fields in an edit are rejected at every level.
    assert [part["additionalProperties"] for part in (edit, edited_action, args)] == [False] * 3


@pytest.mark.parametrize(
    ("answer", "ran_with", "status", "content"),
    [
        ({"type": "approve"}, ["alice"], "success", "sent to alice"),
        (
            {"type": "edit", "edited_action": {"name": "send_email", "args": {"to": "bob"}}},
            ["bob"],
            "success",
            "sent to bob",
        ),
        (
            {"type": "reject", "message": "not now"},
            [],
            "error",
            "User rejected the tool call for `send_email` with reason: not now",
        ),
        ({"type": "respond", "message": "already sent"}, [], "success", "already sent"),
    ],
    ids=["approve", "edit", "reject", "respond"],
)
def test_per_call_resume_with_each_decision(
    answer: Decision, ran_with: list[str], status: str, content: str
) -> None:
    ran: list[str] = []

    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        ran.append(to)
        return f"sent to {to}"

    agent = _agent(
        [send_email],
        [ToolCall(name="send_email", args={"to": "alice"}, id="call_email")],
        {"send_email": True},
    )
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    [intr] = agent.invoke({"messages": [HumanMessage("go")]}, config)["__interrupt__"]

    final = agent.invoke(Command(resume={intr.id: answer}), config)

    [message] = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert (ran, message.status) == (ran_with, status)
    assert str(message.content).endswith(content)
    # Only an edit tells the model that a reviewer replaced the call.
    assert ("Executed instead: send_email" in str(message.content)) == (answer["type"] == "edit")


def test_per_call_edit_leaves_validation_and_injected_args_to_the_tool() -> None:
    @tool
    def send_email(
        to: Annotated[str, AfterValidator(lambda v: f"<{v}>")], runtime: ToolRuntime
    ) -> str:
        """Send an email."""
        return f"sent to {to} for {runtime.tool_call_id}"

    agent = _agent(
        [send_email],
        [ToolCall(name="send_email", args={"to": "alice"}, id="call_email")],
        {"send_email": True},
    )
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    [intr] = agent.invoke({"messages": [HumanMessage("go")]}, config)["__interrupt__"]
    # The reviewer edits only what the model sees; the tool validates the edit once and
    # gets `runtime` injected.
    assert list(intr.response_schema["$defs"]["send_email"]["properties"]) == ["to"]
    edit = {"type": "edit", "edited_action": {"name": "send_email", "args": {"to": "bob"}}}
    final = agent.invoke(Command(resume={intr.id: edit}), config)
    [message] = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert str(message.content).endswith("sent to <bob> for call_email")


def test_per_call_pauses_once_per_gated_call_and_applies_answers_by_id() -> None:
    ran: list[str] = []

    @tool
    def send_email(to: str, cc: str | None = None) -> str:
        """Send an email."""
        ran.append(f"send_email(to={to}, cc={cc})")
        return f"sent to {to}"

    @tool
    def delete_file(path: str) -> str:
        """Delete a file."""
        ran.append(f"delete_file({path})")
        return f"deleted {path}"

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        ran.append(f"read_file({path})")
        return f"contents of {path}"

    calls = [
        ToolCall(name="send_email", args={"to": "alice", "cc": "carol"}, id="call_email"),
        ToolCall(name="delete_file", args={"path": "x.txt"}, id="call_delete"),
        ToolCall(name="delete_file", args={"path": "tmp.txt"}, id="call_delete_tmp"),
        ToolCall(name="read_file", args={"path": "y.txt"}, id="call_read"),
    ]
    agent = _agent(
        [send_email, delete_file, read_file],
        calls,
        {
            "send_email": True,
            "delete_file": {
                "allowed_decisions": ["approve", "reject"],
                "when": lambda req: req.tool_call["args"]["path"] != "tmp.txt",
            },
        },
    )
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    paused = agent.invoke({"messages": [HumanMessage("go")]}, config)["__interrupt__"]
    by_tool = {i.value["name"]: i for i in paused}
    email, delete = by_tool["send_email"], by_tool["delete_file"]
    assert (len(paused), delete.value["args"]) == (2, {"path": "x.txt"})
    assert email.id != delete.id
    # A call that isn't gated, or that `when` lets through, runs without waiting.
    assert sorted(ran) == ["delete_file(tmp.txt)", "read_file(y.txt)"]

    # Each interrupt accepts only its own tool's decisions.
    assert delete.response_schema["oneOf"] == [
        {"$ref": "#/$defs/ApproveDecision"},
        {"$ref": "#/$defs/RejectDecision"},
    ]
    with pytest.raises(ValidationError, match="expected tags: 'approve', 'reject'"):
        agent.invoke(Command(resume={delete.id: {"type": "respond", "message": "x"}}), config)

    # The edit replaces the args: `cc` isn't carried over from the model's call.
    edit = {"type": "edit", "edited_action": {"name": "send_email", "args": {"to": "bob"}}}
    [pending] = agent.invoke(Command(resume={email.id: edit}), config)["__interrupt__"]
    assert pending.id == delete.id

    final = agent.invoke(
        Command(resume={delete.id: {"type": "reject", "message": "keep it"}}), config
    )
    assert sorted(ran) == [  # nothing ran twice
        "delete_file(tmp.txt)",
        "read_file(y.txt)",
        "send_email(to=bob, cc=None)",
    ]
    messages = {m.tool_call_id: m for m in final["messages"] if isinstance(m, ToolMessage)}
    assert 'Executed instead: send_email with arguments {"to": "bob"}' in str(
        messages["call_email"].content
    )
    assert messages["call_delete"].status == "error"
    assert "keep it" in str(messages["call_delete"].content)
    assert "__interrupt__" not in final


@pytest.mark.parametrize(
    ("error", "after"),
    [
        (r"1 validation error for .*\nedit\.edited_action\n", []),
        # Listed before error-handling middleware, HITL's error still reaches the caller.
        (r"edit\.edited_action", [ToolRetryMiddleware(initial_delay=0)]),
        (r"edit\.edited_action", [ToolErrorMiddleware(on_error=lambda *_: "x")]),
    ],
    ids=["hitl_alone", "hitl_wraps_retry", "hitl_wraps_tool_error"],
)
def test_per_call_rejects_a_bad_answer_without_saving_it(
    error: str, after: list[AgentMiddleware]
) -> None:
    ran: list[str] = []

    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        ran.append(to)
        return f"sent to {to}"

    agent = _agent(
        [send_email],
        [ToolCall(name="send_email", args={"to": "alice"}, id="call_email")],
        {"send_email": True},
        *after,
    )
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    [intr] = agent.invoke({"messages": [HumanMessage("go")]}, config)["__interrupt__"]

    with pytest.raises(ValidationError, match=error):
        agent.invoke(Command(resume={intr.id: {"type": "edit"}}), config)
    assert ran == []

    edit = {"type": "edit", "edited_action": {"name": "send_email", "args": {"to": "bob"}}}
    final = agent.invoke(Command(resume={intr.id: edit}), config)
    assert ran == ["bob"]
    assert "__interrupt__" not in final


def test_per_call_needs_a_tool_call_id() -> None:
    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        return f"sent to {to}"

    agent = _agent(
        [send_email],
        [ToolCall(name="send_email", args={"to": "alice"}, id=None)],
        {"send_email": True},
    )
    with pytest.raises(ValueError, match="`send_email` has no ID"):
        agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t"}})


def test_per_call_description_factory_gets_the_graph_runtime() -> None:
    seen: list[object] = []

    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        return f"sent to {to}"

    def describe(tool_call: ToolCall, state: AgentState[Any], runtime: Runtime[ContextT]) -> str:
        seen.append(runtime)
        return f"Email {tool_call['args']['to']}? ({len(state['messages'])} messages)"

    config: InterruptOnConfig = {"allowed_decisions": ["approve"], "description": describe}
    agent = _agent(
        [send_email],
        [ToolCall(name="send_email", args={"to": "alice"}, id="call_email")],
        {"send_email": config},
    )
    result = agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "t"}})
    [intr] = result["__interrupt__"]

    assert intr.value["description"] == "Email alice? (2 messages)"
    assert seen
    assert all(isinstance(runtime, Runtime) for runtime in seen)  # as in batched mode


def test_per_call_same_tool_twice_routes_each_answer_to_its_own_call() -> None:
    ran: list[str] = []

    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        ran.append(to)
        return f"sent to {to}"

    # An empty ID works too, as in batched mode.
    calls = [ToolCall(name="send_email", args={"to": "alice"}, id=i) for i in ("e1", "")]
    agent = _agent([send_email], calls, {"send_email": True})
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    paused = agent.invoke({"messages": [HumanMessage("go")]}, config)["__interrupt__"]
    by_call = {i.value["tool_call_id"]: i.id for i in paused}
    assert set(by_call) == {"e1", ""}

    answers = {by_call["e1"]: {"type": "approve"}, by_call[""]: {"type": "reject"}}
    final = agent.invoke(Command(resume=answers), config)
    assert ran == ["alice"]
    messages = {m.tool_call_id: m for m in final["messages"] if isinstance(m, ToolMessage)}
    assert messages[""].status == "error"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Asyncio context vars require Python 3.11+")
@pytest.mark.parametrize(
    ("answer", "status", "content"),
    [
        (
            {"type": "edit", "edited_action": {"name": "send_email", "args": {"to": "bob"}}},
            "success",
            "sent to bob",
        ),
        ({"type": "reject"}, "error", "The tool was not executed."),
    ],
    ids=["edit", "reject"],
)
async def test_per_call_works_with_ainvoke(answer: Decision, status: str, content: str) -> None:
    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        return f"sent to {to}"

    agent = _agent(
        [send_email],
        [ToolCall(name="send_email", args={"to": "alice"}, id="call_email")],
        {"send_email": True},
    )
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    [intr] = (await agent.ainvoke({"messages": [HumanMessage("go")]}, config))["__interrupt__"]

    with pytest.raises(ValidationError, match=r"edit\.edited_action"):
        await agent.ainvoke(Command(resume={intr.id: {"type": "edit"}}), config)
    final = await agent.ainvoke(Command(resume={intr.id: answer}), config)

    [message] = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert message.status == status
    assert content in str(message.content)


def test_per_call_retried_tool_is_not_reviewed_again_when_hitl_wraps_retry() -> None:
    attempts: list[str] = []

    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        attempts.append(to)
        if len(attempts) == 1:
            msg = "mail server unavailable"
            raise RuntimeError(msg)
        return f"sent to {to}"

    agent = _agent(
        [send_email],
        [ToolCall(name="send_email", args={"to": "alice"}, id="call_email")],
        {"send_email": True},
        ToolRetryMiddleware(initial_delay=0),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    [intr] = agent.invoke({"messages": [HumanMessage("go")]}, config)["__interrupt__"]
    final = agent.invoke(Command(resume={intr.id: {"type": "approve"}}), config)

    assert "__interrupt__" not in final  # the retry didn't ask the reviewer again
    assert attempts == ["alice", "alice"]
    [message] = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert message.content == "sent to alice"
