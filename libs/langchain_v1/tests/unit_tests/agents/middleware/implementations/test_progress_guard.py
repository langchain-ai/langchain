"""Tests for ProgressGuardMiddleware."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool

from langchain.agents.factory import create_agent
from langchain.agents.middleware import (
    AgentProgressStalledError,
    ProgressGuardMiddleware,
    wrap_tool_call,
)
from tests.unit_tests.agents.model import FakeToolCallingModel


class FailureCounter:
    """Track tool invocations while returning controlled failures."""

    def __init__(self) -> None:
        self.calls = 0

    def fail(self, value: str) -> str:
        """Always raise a deterministic failure for the given value."""
        self.calls += 1
        msg = f"backend failure for {value}"
        raise ValueError(msg)

    def fail_with_request_id(self, value: str) -> str:
        """Raise a failure whose request ID changes on each call."""
        self.calls += 1
        msg = f"backend failure for {value} request_id=req-{self.calls}"
        raise ValueError(msg)


@tool
def working_tool(value: str) -> str:
    """Return a successful value."""
    return f"success: {value}"


@tool
def other_working_tool(value: str) -> str:
    """Return another successful value."""
    return f"other success: {value}"


def test_progress_guard_initialization_defaults() -> None:
    """Test default initialization values."""
    guard = ProgressGuardMiddleware()

    assert guard.max_consecutive_identical_steps == 3
    assert guard.exit_behavior == "error"
    assert guard.catch_tool_exceptions is False
    assert guard.failure_only is True
    assert guard.tools == []
    assert guard._tool_filter is None


def test_progress_guard_invalid_initialization() -> None:
    """Test initialization validation."""
    with pytest.raises(ValueError, match="max_consecutive_identical_steps must be >= 2"):
        ProgressGuardMiddleware(max_consecutive_identical_steps=1)

    with pytest.raises(ValueError, match="exit_behavior must be 'error' or 'end'"):
        ProgressGuardMiddleware(exit_behavior="continue")  # type: ignore[arg-type]


def test_progress_guard_ignores_repeated_successful_tool_output_by_default() -> None:
    """Test repeated successful tool exchanges are ignored by default."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-2")],
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-3")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[working_tool],
        middleware=[ProgressGuardMiddleware(max_consecutive_identical_steps=3)],
    )

    result = agent.invoke({"messages": [HumanMessage("keep trying")]})

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 3
    assert model.index == 4


def test_progress_guard_can_detect_repeated_successful_tool_output() -> None:
    """Test repeated successful tool exchanges can be treated as no progress."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-2")],
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-3")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[working_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=3,
                failure_only=False,
            )
        ],
    )

    with pytest.raises(AgentProgressStalledError, match="no_progress_detected") as exc_info:
        agent.invoke({"messages": [HumanMessage("keep trying")]})

    assert exc_info.value.reason == "no_progress_detected"
    assert exc_info.value.consecutive_steps == 3
    assert model.index == 3


def test_progress_guard_detects_repeated_error_messages_from_other_middleware() -> None:
    """Test repeated error ToolMessages are detected without catching exceptions."""

    @wrap_tool_call
    def return_error_message(request: Any, handler: Any) -> ToolMessage:
        """Short-circuit tool execution with an error ToolMessage."""
        return ToolMessage(
            content="backend unavailable",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-2")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[working_tool],
        middleware=[
            ProgressGuardMiddleware(max_consecutive_identical_steps=2),
            return_error_message,
        ],
    )

    with pytest.raises(AgentProgressStalledError, match="no_progress_detected"):
        agent.invoke({"messages": [HumanMessage("keep trying")]})

    assert model.index == 2


def test_progress_guard_detects_repeated_multi_tool_exchange() -> None:
    """Test repeated identical multi-tool exchanges are treated as no progress."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="working_tool", args={"value": "same"}, id="call-1"),
                ToolCall(name="other_working_tool", args={"value": "same"}, id="call-2"),
            ],
            [
                ToolCall(name="working_tool", args={"value": "same"}, id="call-3"),
                ToolCall(name="other_working_tool", args={"value": "same"}, id="call-4"),
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[working_tool, other_working_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=2,
                failure_only=False,
            )
        ],
    )

    with pytest.raises(AgentProgressStalledError, match="no_progress_detected"):
        agent.invoke({"messages": [HumanMessage("keep trying")]})

    assert model.index == 2


def test_progress_guard_can_preserve_raw_tool_exception_behavior() -> None:
    """Test raw tool exceptions still abort when error handling is disabled."""
    failure_counter = FailureCounter()

    @tool
    def failing_tool(value: str) -> str:
        """Always fail with the same error."""
        return failure_counter.fail(value)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="failing_tool", args={"value": "same"}, id="call-2")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[ProgressGuardMiddleware(max_consecutive_identical_steps=2)],
    )

    with pytest.raises(ValueError, match="backend failure for same"):
        agent.invoke({"messages": [HumanMessage("keep trying")]})

    assert failure_counter.calls == 1
    assert model.index == 1


def test_progress_guard_can_catch_tool_exceptions() -> None:
    """Test repeated caught tool exceptions are treated as no progress."""
    failure_counter = FailureCounter()

    @tool
    def failing_tool(value: str) -> str:
        """Always fail with the same error."""
        return failure_counter.fail(value)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="failing_tool", args={"value": "same"}, id="call-2")],
            [ToolCall(name="failing_tool", args={"value": "same"}, id="call-3")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=3,
                catch_tool_exceptions=True,
            )
        ],
    )

    with pytest.raises(AgentProgressStalledError, match="no_progress_detected"):
        agent.invoke({"messages": [HumanMessage("keep trying")]})

    assert failure_counter.calls == 3
    assert model.index == 3


def test_progress_guard_end_behavior_appends_final_ai_message() -> None:
    """Test graceful termination appends a final AI message."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-2")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[working_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=2,
                exit_behavior="end",
                failure_only=False,
            )
        ],
    )

    result = agent.invoke({"messages": [HumanMessage("keep trying")]})

    assert result["messages"][-1].content.startswith(
        "Agent stopped because no_progress_detected"
    )
    assert model.index == 2


def test_progress_guard_normalizes_volatile_output_details() -> None:
    """Test changing request IDs do not break equivalent exchange detection."""
    failure_counter = FailureCounter()

    @tool
    def failing_tool(value: str) -> str:
        """Always fail with a changing request ID."""
        return failure_counter.fail_with_request_id(value)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="failing_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="failing_tool", args={"value": "same"}, id="call-2")],
            [ToolCall(name="failing_tool", args={"value": "same"}, id="call-3")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=3,
                catch_tool_exceptions=True,
            )
        ],
    )

    with pytest.raises(AgentProgressStalledError, match="no_progress_detected"):
        agent.invoke({"messages": [HumanMessage("keep trying")]})

    assert failure_counter.calls == 3


def test_progress_guard_resets_when_args_change() -> None:
    """Test changed tool arguments break the no-progress streak."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "one"}, id="call-1")],
            [ToolCall(name="working_tool", args={"value": "one"}, id="call-2")],
            [ToolCall(name="working_tool", args={"value": "two"}, id="call-3")],
            [ToolCall(name="working_tool", args={"value": "one"}, id="call-4")],
            [ToolCall(name="working_tool", args={"value": "one"}, id="call-5")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[working_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=3,
                failure_only=False,
            )
        ],
    )

    result = agent.invoke({"messages": [HumanMessage("keep trying")]})

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 5
    assert model.index == 6


def test_progress_guard_resets_when_tool_name_changes() -> None:
    """Test changed tool name breaks the no-progress streak."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="other_working_tool", args={"value": "same"}, id="call-2")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[working_tool, other_working_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=2,
                failure_only=False,
            )
        ],
    )

    result = agent.invoke({"messages": [HumanMessage("keep trying")]})

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 2
    assert model.index == 3


def test_progress_guard_resets_when_output_changes() -> None:
    """Test changed tool output breaks the no-progress streak."""
    calls = {"count": 0}

    @tool
    def changing_output_tool(value: str) -> str:
        """Return a different message on each call."""
        calls["count"] += 1
        return f"success: {value} {calls['count']}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="changing_output_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="changing_output_tool", args={"value": "same"}, id="call-2")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[changing_output_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=2,
                failure_only=False,
            )
        ],
    )

    result = agent.invoke({"messages": [HumanMessage("keep trying")]})

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 2
    assert model.index == 3


def test_progress_guard_resets_when_status_changes() -> None:
    """Test changed output status breaks the no-progress streak."""
    calls = {"count": 0}

    @wrap_tool_call
    def changing_status_middleware(request: Any, handler: Any) -> ToolMessage:
        calls["count"] += 1
        status = "error" if calls["count"] == 1 else "success"
        return ToolMessage(
            content="same output",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status=status,
        )

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-2")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[working_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=2,
                failure_only=False,
            ),
            changing_status_middleware,
        ],
    )

    result = agent.invoke({"messages": [HumanMessage("keep trying")]})

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 2
    assert model.index == 3


def test_progress_guard_resets_on_unmonitored_tool() -> None:
    """Test an unmonitored tool exchange resets progress tracking."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="other_working_tool", args={"value": "same"}, id="call-2")],
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-3")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[working_tool, other_working_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=2,
                tools=["working_tool"],
                failure_only=False,
            )
        ],
    )

    result = agent.invoke({"messages": [HumanMessage("keep trying")]})

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 3
    assert model.index == 4


def test_progress_guard_resets_on_missing_tool_result() -> None:
    """Test an incomplete tool exchange resets progress tracking."""
    guard = ProgressGuardMiddleware(max_consecutive_identical_steps=2)
    state = {
        "messages": [
            AIMessage(
                content="",
                id="ai-1",
                tool_calls=[ToolCall(name="working_tool", args={"value": "same"}, id="call-1")],
            )
        ],
        "run_progress_guard_initialized": True,
        "run_progress_guard_last_processed_exchange_id": "previous",
        "run_progress_guard_last_signature": "previous-signature",
        "run_progress_guard_consecutive_count": 1,
    }

    update = guard.before_model(state, runtime=None)  # type: ignore[arg-type]

    assert update == {
        "run_progress_guard_last_processed_exchange_id": "ai-1",
        "run_progress_guard_last_signature": None,
        "run_progress_guard_consecutive_count": 0,
    }


def test_progress_guard_ignores_preexisting_history_on_run_start() -> None:
    """Test existing thread history is baselined instead of counted immediately."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "same"}, id="new-call-1")],
            [],
        ]
    )

    prior_messages: list[Any] = [
        HumanMessage("Earlier attempt"),
        AIMessage(
            content="",
            id="prior-ai",
            tool_calls=[ToolCall(name="working_tool", args={"value": "same"}, id="prior-call")],
        ),
        ToolMessage(
            content="success: same",
            tool_call_id="prior-call",
            name="working_tool",
            status="success",
        ),
        HumanMessage("Try again"),
    ]

    agent = create_agent(
        model=model,
        tools=[working_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=2,
                failure_only=False,
            )
        ],
    )

    result = agent.invoke({"messages": prior_messages})

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 2
    assert model.index == 2


async def test_progress_guard_async_parity() -> None:
    """Test async execution detects repeated no-progress exchanges."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-1")],
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-2")],
            [ToolCall(name="working_tool", args={"value": "same"}, id="call-3")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[working_tool],
        middleware=[
            ProgressGuardMiddleware(
                max_consecutive_identical_steps=3,
                failure_only=False,
            )
        ],
    )

    with pytest.raises(AgentProgressStalledError, match="no_progress_detected"):
        await agent.ainvoke({"messages": [HumanMessage("keep trying")]})

    assert model.index == 3
