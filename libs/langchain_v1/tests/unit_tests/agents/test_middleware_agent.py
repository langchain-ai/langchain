import warnings
from collections.abc import Awaitable, Callable
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolCall,
    ToolMessage,
)
from typing import cast

from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field
from syrupy.assertion import SnapshotAssertion
from typing_extensions import Annotated

from langchain.agents.middleware.human_in_the_loop import (
    Action,
    HumanInTheLoopMiddleware,
)
from langchain.agents.middleware.planning import (
    PlanningMiddleware,
    PlanningState,
    WRITE_TODOS_SYSTEM_PROMPT,
    write_todos,
    WRITE_TODOS_TOOL_DESCRIPTION,
)
from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitMiddleware,
    ModelCallLimitExceededError,
)
from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
from langchain.agents.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    hook_config,
    ModelRequest,
    OmitFromInput,
    OmitFromOutput,
    PrivateStateAttr,
)
from langchain.agents.factory import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import InjectedState

from .messages import _AnyIdHumanMessage, _AnyIdToolMessage
from .model import FakeToolCallingModel


def test_create_agent_diagram(
    snapshot: SnapshotAssertion,
):
    class NoopOne(AgentMiddleware):
        def before_model(self, state, runtime):
            pass

    class NoopTwo(AgentMiddleware):
        def before_model(self, state, runtime):
            pass

    class NoopThree(AgentMiddleware):
        def before_model(self, state, runtime):
            pass

    class NoopFour(AgentMiddleware):
        def after_model(self, state, runtime):
            pass

    class NoopFive(AgentMiddleware):
        def after_model(self, state, runtime):
            pass

    class NoopSix(AgentMiddleware):
        def after_model(self, state, runtime):
            pass

    class NoopSeven(AgentMiddleware):
        def before_model(self, state, runtime):
            pass

        def after_model(self, state, runtime):
            pass

    class NoopEight(AgentMiddleware):
        def before_model(self, state, runtime):
            pass

        def after_model(self, state, runtime):
            pass

    class NoopNine(AgentMiddleware):
        def before_model(self, state, runtime):
            pass

        def after_model(self, state, runtime):
            pass

    class NoopTen(AgentMiddleware):
        def before_model(self, state, runtime):
            pass

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            return handler(request)

        def after_model(self, state, runtime):
            pass

    class NoopEleven(AgentMiddleware):
        def before_model(self, state, runtime):
            pass

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            return handler(request)

        def after_model(self, state, runtime):
            pass

    agent_zero = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
    )

    assert agent_zero.get_graph().draw_mermaid() == snapshot

    agent_one = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne()],
    )

    assert agent_one.get_graph().draw_mermaid() == snapshot

    agent_two = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne(), NoopTwo()],
    )

    assert agent_two.get_graph().draw_mermaid() == snapshot

    agent_three = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne(), NoopTwo(), NoopThree()],
    )

    assert agent_three.get_graph().draw_mermaid() == snapshot

    agent_four = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour()],
    )

    assert agent_four.get_graph().draw_mermaid() == snapshot

    agent_five = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour(), NoopFive()],
    )

    assert agent_five.get_graph().draw_mermaid() == snapshot

    agent_six = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour(), NoopFive(), NoopSix()],
    )

    assert agent_six.get_graph().draw_mermaid() == snapshot

    agent_seven = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven()],
    )

    assert agent_seven.get_graph().draw_mermaid() == snapshot

    agent_eight = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight()],
    )

    assert agent_eight.get_graph().draw_mermaid() == snapshot

    agent_nine = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight(), NoopNine()],
    )

    assert agent_nine.get_graph().draw_mermaid() == snapshot

    agent_ten = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopTen()],
    )

    assert agent_ten.get_graph().draw_mermaid() == snapshot

    agent_eleven = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopTen(), NoopEleven()],
    )

    assert agent_eleven.get_graph().draw_mermaid() == snapshot


def test_create_agent_invoke(
    snapshot: SnapshotAssertion,
    sync_checkpointer: BaseCheckpointSaver,
):
    calls = []

    class NoopSeven(AgentMiddleware):
        def before_model(self, state, runtime):
            calls.append("NoopSeven.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("NoopSeven.wrap_model_call")
            return handler(request)

        def after_model(self, state, runtime):
            calls.append("NoopSeven.after_model")

    class NoopEight(AgentMiddleware):
        def before_model(self, state, runtime):
            calls.append("NoopEight.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("NoopEight.wrap_model_call")
            return handler(request)

        def after_model(self, state, runtime):
            calls.append("NoopEight.after_model")

    @tool
    def my_tool(input: str) -> str:
        """A great tool"""
        calls.append("my_tool")
        return input.upper()

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {"args": {"input": "yo"}, "id": "1", "name": "my_tool"},
                ],
                [],
            ]
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight()],
        checkpointer=sync_checkpointer,
    )

    thread1 = {"configurable": {"thread_id": "1"}}
    assert agent_one.invoke({"messages": ["hello"]}, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            AIMessage(
                content="You are a helpful assistant.-hello",
                additional_kwargs={},
                response_metadata={},
                id="0",
                tool_calls=[
                    {
                        "name": "my_tool",
                        "args": {"input": "yo"},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(content="YO", name="my_tool", tool_call_id="1"),
            AIMessage(
                content="You are a helpful assistant.-hello-You are a helpful assistant.-hello-YO",
                additional_kwargs={},
                response_metadata={},
                id="1",
            ),
        ],
    }
    assert calls == [
        "NoopSeven.before_model",
        "NoopEight.before_model",
        "NoopSeven.wrap_model_call",
        "NoopEight.wrap_model_call",
        "NoopEight.after_model",
        "NoopSeven.after_model",
        "my_tool",
        "NoopSeven.before_model",
        "NoopEight.before_model",
        "NoopSeven.wrap_model_call",
        "NoopEight.wrap_model_call",
        "NoopEight.after_model",
        "NoopSeven.after_model",
    ]


def test_create_agent_jump(
    snapshot: SnapshotAssertion,
    sync_checkpointer: BaseCheckpointSaver,
):
    calls = []

    class NoopSeven(AgentMiddleware):
        def before_model(self, state, runtime):
            calls.append("NoopSeven.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("NoopSeven.wrap_model_call")
            return handler(request)

        def after_model(self, state, runtime):
            calls.append("NoopSeven.after_model")

    class NoopEight(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        def before_model(self, state, runtime) -> dict[str, Any]:
            calls.append("NoopEight.before_model")
            return {"jump_to": "end"}

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("NoopEight.wrap_model_call")
            return handler(request)

        def after_model(self, state, runtime):
            calls.append("NoopEight.after_model")

    @tool
    def my_tool(input: str) -> str:
        """A great tool"""
        calls.append("my_tool")
        return input.upper()

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[ToolCall(id="1", name="my_tool", args={"input": "yo"})]],
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight()],
        checkpointer=sync_checkpointer,
    )

    if isinstance(sync_checkpointer, InMemorySaver):
        assert agent_one.get_graph().draw_mermaid() == snapshot

    thread1 = {"configurable": {"thread_id": "1"}}
    assert agent_one.invoke({"messages": []}, thread1) == {"messages": []}
    assert calls == ["NoopSeven.before_model", "NoopEight.before_model"]


def test_simple_agent_graph(snapshot: SnapshotAssertion) -> None:
    @tool
    def my_tool(input_string: str) -> str:
        """A great tool."""
        return input_string

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[ToolCall(id="1", name="my_tool", args={"input": "yo"})]],
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
    )

    assert agent_one.get_graph().draw_mermaid() == snapshot


def test_agent_graph_with_jump_to_end_as_after_agent(snapshot: SnapshotAssertion) -> None:
    @tool
    def my_tool(input_string: str) -> str:
        """A great tool."""
        return input_string

    class NoopZero(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        def before_agent(self, state, runtime) -> None:
            return None

    class NoopOne(AgentMiddleware):
        def after_agent(self, state, runtime) -> None:
            return None

    class NoopTwo(AgentMiddleware):
        def after_agent(self, state, runtime) -> None:
            return None

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[ToolCall(id="1", name="my_tool", args={"input": "yo"})]],
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopZero(), NoopOne(), NoopTwo()],
    )

    assert agent_one.get_graph().draw_mermaid() == snapshot


# Tests for HumanInTheLoopMiddleware
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
    state: dict[str, Any] = {"messages": []}
    result = middleware.after_model(state, None)
    assert result is None

    # Test with message but no tool calls
    state = {"messages": [HumanMessage(content="Hello"), AIMessage(content="Hi there")]}
    result = middleware.after_model(state, None)
    assert result is None

    # Test with tool calls that don't require interrupts
    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "other_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}
    result = middleware.after_model(state, None)
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
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_accept(requests):
        return {"decisions": [{"type": "approve"}]}

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_accept):
        result = middleware.after_model(state, None)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0] == ai_message
        assert result["messages"][0].tool_calls == ai_message.tool_calls

    state["messages"].append(
        ToolMessage(content="Tool message", name="test_tool", tool_call_id="1")
    )
    state["messages"].append(AIMessage(content="test_tool called with result: Tool message"))

    result = middleware.after_model(state, None)
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
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_edit(requests):
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="test_tool",
                        arguments={"input": "edited"},
                    ),
                }
            ]
        }

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_edit):
        result = middleware.after_model(state, None)
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
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_response(requests):
        return {"decisions": [{"type": "reject", "message": "Custom response message"}]}

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_response
    ):
        result = middleware.after_model(state, None)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], AIMessage)
        assert isinstance(result["messages"][1], ToolMessage)
        assert result["messages"][1].content == "Custom response message"
        assert result["messages"][1].name == "test_tool"
        assert result["messages"][1].tool_call_id == "1"


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
    state = {"messages": [HumanMessage(content="What's the weather?"), ai_message]}

    def mock_mixed_responses(requests):
        return {
            "decisions": [
                {"type": "approve"},
                {"type": "reject", "message": "User rejected this tool call"},
            ]
        }

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_mixed_responses
    ):
        result = middleware.after_model(state, None)
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
    state = {"messages": [HumanMessage(content="What's the weather?"), ai_message]}

    def mock_edit_responses(requests):
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="get_forecast",
                        arguments={"location": "New York"},
                    ),
                },
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="get_temperature",
                        arguments={"location": "New York"},
                    ),
                },
            ]
        }

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_edit_responses
    ):
        result = middleware.after_model(state, None)
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
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_edit_with_args(requests):
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="test_tool",
                        arguments={"input": "modified"},
                    ),
                }
            ]
        }

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        side_effect=mock_edit_with_args,
    ):
        result = middleware.after_model(state, None)
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
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_unknown(requests):
        return {"decisions": [{"type": "unknown"}]}

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_unknown):
        with pytest.raises(
            ValueError,
            match=r"Unexpected human decision: {'type': 'unknown'}. Decision type 'unknown' is not allowed for tool 'test_tool'. Expected one of \['approve', 'edit', 'reject'\] based on the tool's configuration.",
        ):
            middleware.after_model(state, None)


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
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_disallowed_action(requests):
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="test_tool",
                        arguments={"input": "modified"},
                    ),
                }
            ]
        }

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        side_effect=mock_disallowed_action,
    ):
        with pytest.raises(
            ValueError,
            match=r"Unexpected human decision: {'type': 'edit', 'edited_action': {'name': 'test_tool', 'arguments': {'input': 'modified'}}}. Decision type 'edit' is not allowed for tool 'test_tool'. Expected one of \['approve', 'reject'\] based on the tool's configuration.",
        ):
            middleware.after_model(state, None)


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
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_accept(requests):
        return {"decisions": [{"type": "approve"}]}

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_accept):
        result = middleware.after_model(state, None)
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
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    captured_request = None

    def mock_capture_requests(request):
        nonlocal captured_request
        captured_request = request
        return {"decisions": [{"type": "approve"}]}

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_capture_requests
    ):
        middleware.after_model(state, None)

        assert captured_request is not None
        assert "action_requests" in captured_request
        assert "review_configs" in captured_request

        assert len(captured_request["action_requests"]) == 1
        action_request = captured_request["action_requests"][0]
        assert action_request["name"] == "test_tool"
        assert action_request["arguments"] == {"input": "test", "location": "SF"}

        assert len(captured_request["review_configs"]) == 1
        review_config = captured_request["review_configs"][0]
        assert review_config["action_name"] == "test_tool"
        assert review_config["allowed_decisions"] == ["approve", "edit", "reject"]
        assert "Custom prefix" in review_config["description"]
        assert "Tool: test_tool" in review_config["description"]
        assert "Args: {'input': 'test', 'location': 'SF'}" in review_config["description"]


def test_human_in_the_loop_middleware_boolean_configs() -> None:
    """Test HITL middleware with boolean tool configs."""
    middleware = HumanInTheLoopMiddleware(interrupt_on={"test_tool": True})

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    # Test accept
    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": [{"type": "approve"}]},
    ):
        result = middleware.after_model(state, None)
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
                        arguments={"input": "edited"},
                    ),
                }
            ]
        },
    ):
        result = middleware.after_model(state, None)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls[0]["args"] == {"input": "edited"}

    middleware = HumanInTheLoopMiddleware(interrupt_on={"test_tool": False})

    result = middleware.after_model(state, None)
    # No interruption should occur
    assert result is None


def test_human_in_the_loop_middleware_sequence_mismatch() -> None:
    """Test that sequence mismatch in resume raises an error."""
    middleware = HumanInTheLoopMiddleware(interrupt_on={"test_tool": True})

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    # Test with too few responses
    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={"decisions": []},  # No responses for 1 tool call
    ):
        with pytest.raises(
            ValueError,
            match=r"Number of human decisions \(0\) does not match number of hanging tool calls \(1\)\.",
        ):
            middleware.after_model(state, None)

    # Test with too many responses
    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value={
            "decisions": [
                {"type": "approve"},
                {"type": "approve"},
            ]
        },  # 2 responses for 1 tool call
    ):
        with pytest.raises(
            ValueError,
            match=r"Number of human decisions \(2\) does not match number of hanging tool calls \(1\)\.",
        ):
            middleware.after_model(state, None)


def test_human_in_the_loop_middleware_description_as_callable() -> None:
    """Test that description field accepts both string and callable."""

    def custom_description(tool_call: ToolCall, state: AgentState, runtime: Runtime) -> str:
        """Generate a custom description."""
        return f"Custom: {tool_call['name']} with args {tool_call['args']}"

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "tool_with_callable": {
                "allowed_decisions": ["approve"],
                "description": custom_description,
            },
            "tool_with_string": {
                "allowed_decisions": ["approve"],
                "description": "Static description",
            },
        }
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[
            {"name": "tool_with_callable", "args": {"x": 1}, "id": "1"},
            {"name": "tool_with_string", "args": {"y": 2}, "id": "2"},
        ],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    captured_request = None

    def mock_capture_requests(request):
        nonlocal captured_request
        captured_request = request
        return {"decisions": [{"type": "approve"}, {"type": "approve"}]}

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_capture_requests
    ):
        middleware.after_model(state, None)

        assert captured_request is not None
        assert "review_configs" in captured_request
        assert len(captured_request["review_configs"]) == 2

        # Check callable description
        assert (
            captured_request["review_configs"][0]["description"]
            == "Custom: tool_with_callable with args {'x': 1}"
        )

        # Check string description
        assert captured_request["review_configs"][1]["description"] == "Static description"


# Tests for AnthropicPromptCachingMiddleware
def test_anthropic_prompt_caching_middleware_initialization() -> None:
    """Test AnthropicPromptCachingMiddleware initialization."""
    # Test with custom values
    middleware = AnthropicPromptCachingMiddleware(
        type="ephemeral", ttl="1h", min_messages_to_cache=5
    )
    assert middleware.type == "ephemeral"
    assert middleware.ttl == "1h"
    assert middleware.min_messages_to_cache == 5

    # Test with default values
    middleware = AnthropicPromptCachingMiddleware()
    assert middleware.type == "ephemeral"
    assert middleware.ttl == "5m"
    assert middleware.min_messages_to_cache == 0

    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response", **req.model_settings)

    result = middleware.wrap_model_call(fake_request, mock_handler)
    # Check that model_settings were passed through via the request
    assert fake_request.model_settings == {"cache_control": {"type": "ephemeral", "ttl": "5m"}}


def test_anthropic_prompt_caching_middleware_unsupported_model() -> None:
    """Test AnthropicPromptCachingMiddleware with unsupported model."""
    from typing import cast

    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": [HumanMessage("Hello")]},
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="raise")

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    with pytest.raises(
        ValueError,
        match="AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models. Please install langchain-anthropic.",
    ):
        middleware.wrap_model_call(fake_request, mock_handler)

    langchain_anthropic = ModuleType("langchain_anthropic")

    class MockChatAnthropic:
        pass

    langchain_anthropic.ChatAnthropic = MockChatAnthropic

    with patch.dict("sys.modules", {"langchain_anthropic": langchain_anthropic}):
        with pytest.raises(
            ValueError,
            match="AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models, not instances of",
        ):
            middleware.wrap_model_call(fake_request, mock_handler)

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="warn")

    with warnings.catch_warnings(record=True) as w:
        result = middleware.wrap_model_call(fake_request, mock_handler)
        assert len(w) == 1
        assert (
            "AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models. Please install langchain-anthropic."
            in str(w[-1].message)
        )
        assert isinstance(result, AIMessage)

    with warnings.catch_warnings(record=True) as w:
        with patch.dict("sys.modules", {"langchain_anthropic": langchain_anthropic}):
            result = middleware.wrap_model_call(fake_request, mock_handler)
            assert isinstance(result, AIMessage)
            assert len(w) == 1
            assert (
                "AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models, not instances of"
                in str(w[-1].message)
            )

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")

    result = middleware.wrap_model_call(fake_request, mock_handler)
    assert isinstance(result, AIMessage)

    with patch.dict("sys.modules", {"langchain_anthropic": {"ChatAnthropic": object()}}):
        result = middleware.wrap_model_call(fake_request, mock_handler)
        assert isinstance(result, AIMessage)


# Tests for SummarizationMiddleware
def test_summarization_middleware_initialization() -> None:
    """Test SummarizationMiddleware initialization."""
    model = FakeToolCallingModel()
    middleware = SummarizationMiddleware(
        model=model,
        max_tokens_before_summary=1000,
        messages_to_keep=10,
        summary_prompt="Custom prompt: {messages}",
        summary_prefix="Custom prefix:",
    )

    assert middleware.model == model
    assert middleware.max_tokens_before_summary == 1000
    assert middleware.messages_to_keep == 10
    assert middleware.summary_prompt == "Custom prompt: {messages}"
    assert middleware.summary_prefix == "Custom prefix:"

    # Test with string model
    with patch(
        "langchain.agents.middleware.summarization.init_chat_model",
        return_value=FakeToolCallingModel(),
    ):
        middleware = SummarizationMiddleware(model="fake-model")
        assert isinstance(middleware.model, FakeToolCallingModel)


def test_summarization_middleware_no_summarization_cases() -> None:
    """Test SummarizationMiddleware when summarization is not needed or disabled."""
    model = FakeToolCallingModel()
    middleware = SummarizationMiddleware(model=model, max_tokens_before_summary=1000)

    # Test when summarization is disabled
    middleware_disabled = SummarizationMiddleware(model=model, max_tokens_before_summary=None)
    state = {"messages": [HumanMessage(content="Hello"), AIMessage(content="Hi")]}
    result = middleware_disabled.before_model(state, None)
    assert result is None

    # Test when token count is below threshold
    def mock_token_counter(messages):
        return 500  # Below threshold

    middleware.token_counter = mock_token_counter
    result = middleware.before_model(state, None)
    assert result is None


def test_summarization_middleware_helper_methods() -> None:
    """Test SummarizationMiddleware helper methods."""
    model = FakeToolCallingModel()
    middleware = SummarizationMiddleware(model=model, max_tokens_before_summary=1000)

    # Test message ID assignment
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi")]
    middleware._ensure_message_ids(messages)
    for msg in messages:
        assert msg.id is not None

    # Test message partitioning
    messages = [
        HumanMessage(content="1"),
        HumanMessage(content="2"),
        HumanMessage(content="3"),
        HumanMessage(content="4"),
        HumanMessage(content="5"),
    ]
    to_summarize, preserved = middleware._partition_messages(messages, 2)
    assert len(to_summarize) == 2
    assert len(preserved) == 3
    assert to_summarize == messages[:2]
    assert preserved == messages[2:]

    # Test summary message building
    summary = "This is a test summary"
    new_messages = middleware._build_new_messages(summary)
    assert len(new_messages) == 1
    assert isinstance(new_messages[0], HumanMessage)
    assert "Here is a summary of the conversation to date:" in new_messages[0].content
    assert summary in new_messages[0].content

    # Test tool call detection
    ai_message_no_tools = AIMessage(content="Hello")
    assert not middleware._has_tool_calls(ai_message_no_tools)

    ai_message_with_tools = AIMessage(
        content="Hello", tool_calls=[{"name": "test", "args": {}, "id": "1"}]
    )
    assert middleware._has_tool_calls(ai_message_with_tools)

    human_message = HumanMessage(content="Hello")
    assert not middleware._has_tool_calls(human_message)


def test_summarization_middleware_tool_call_safety() -> None:
    """Test SummarizationMiddleware tool call safety logic."""
    model = FakeToolCallingModel()
    middleware = SummarizationMiddleware(
        model=model, max_tokens_before_summary=1000, messages_to_keep=3
    )

    # Test safe cutoff point detection with tool calls
    messages = [
        HumanMessage(content="1"),
        AIMessage(content="2", tool_calls=[{"name": "test", "args": {}, "id": "1"}]),
        ToolMessage(content="3", tool_call_id="1"),
        HumanMessage(content="4"),
    ]

    # Safe cutoff (doesn't separate AI/Tool pair)
    is_safe = middleware._is_safe_cutoff_point(messages, 0)
    assert is_safe is True

    # Unsafe cutoff (separates AI/Tool pair)
    is_safe = middleware._is_safe_cutoff_point(messages, 2)
    assert is_safe is False

    # Test tool call ID extraction
    ids = middleware._extract_tool_call_ids(messages[1])
    assert ids == {"1"}


def test_summarization_middleware_summary_creation() -> None:
    """Test SummarizationMiddleware summary creation."""

    class MockModel(BaseChatModel):
        def invoke(self, prompt):
            return AIMessage(content="Generated summary")

        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        @property
        def _llm_type(self):
            return "mock"

    middleware = SummarizationMiddleware(model=MockModel(), max_tokens_before_summary=1000)

    # Test normal summary creation
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi")]
    summary = middleware._create_summary(messages)
    assert summary == "Generated summary"

    # Test empty messages
    summary = middleware._create_summary([])
    assert summary == "No previous conversation history."

    # Test error handling
    class ErrorModel(BaseChatModel):
        def invoke(self, prompt):
            raise Exception("Model error")

        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        @property
        def _llm_type(self):
            return "mock"

    middleware_error = SummarizationMiddleware(model=ErrorModel(), max_tokens_before_summary=1000)
    summary = middleware_error._create_summary(messages)
    assert "Error generating summary: Model error" in summary


def test_summarization_middleware_full_workflow() -> None:
    """Test SummarizationMiddleware complete summarization workflow."""

    class MockModel(BaseChatModel):
        def invoke(self, prompt):
            return AIMessage(content="Generated summary")

        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        @property
        def _llm_type(self):
            return "mock"

    middleware = SummarizationMiddleware(
        model=MockModel(), max_tokens_before_summary=1000, messages_to_keep=2
    )

    # Mock high token count to trigger summarization
    def mock_token_counter(messages):
        return 1500  # Above threshold

    middleware.token_counter = mock_token_counter

    messages = [
        HumanMessage(content="1"),
        HumanMessage(content="2"),
        HumanMessage(content="3"),
        HumanMessage(content="4"),
        HumanMessage(content="5"),
    ]

    state = {"messages": messages}
    result = middleware.before_model(state, None)

    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Should have RemoveMessage for cleanup
    assert isinstance(result["messages"][0], RemoveMessage)
    assert result["messages"][0].id == REMOVE_ALL_MESSAGES

    # Should have summary message
    summary_message = None
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage) and "summary of the conversation" in msg.content:
            summary_message = msg
            break

    assert summary_message is not None
    assert "Generated summary" in summary_message.content


def test_on_model_call() -> None:
    class ModifyMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            request.messages.append(HumanMessage("remember to be nice!"))
            return handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[ModifyMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert result["messages"][0].content == "Hello"
    assert result["messages"][1].content == "remember to be nice!"
    assert (
        result["messages"][2].content == "You are a helpful assistant.-Hello-remember to be nice!"
    )


def test_tools_to_model_edge_with_structured_and_regular_tool_calls():
    """Test that when there are both structured and regular tool calls, we execute regular and jump to END."""

    class WeatherResponse(BaseModel):
        """Weather response."""

        temperature: float = Field(description="Temperature in fahrenheit")
        condition: str = Field(description="Weather condition")

    @tool
    def regular_tool(query: str) -> str:
        """A regular tool that returns a string."""
        return f"Regular tool result for: {query}"

    # Create a fake model that returns both structured and regular tool calls
    class FakeModelWithBothToolCalls(FakeToolCallingModel):
        def __init__(self):
            super().__init__()
            self.tool_calls = [
                [
                    ToolCall(
                        name="WeatherResponse",
                        args={"temperature": 72.0, "condition": "sunny"},
                        id="structured_call_1",
                    ),
                    ToolCall(
                        name="regular_tool", args={"query": "test query"}, id="regular_call_1"
                    ),
                ]
            ]

    # Create agent with both structured output and regular tools
    agent = create_agent(
        model=FakeModelWithBothToolCalls(),
        tools=[regular_tool],
        response_format=ToolStrategy(schema=WeatherResponse),
    )

    # Invoke the agent (already compiled)
    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather and help me with a query?")]}
    )

    # Verify that we have the expected messages:
    # 1. Human message
    # 2. AI message with both tool calls
    # 3. Tool message from structured tool call
    # 4. Tool message from regular tool call

    messages = result["messages"]
    assert len(messages) >= 4

    # Check that we have the AI message with both tool calls
    ai_message = messages[1]
    assert isinstance(ai_message, AIMessage)
    assert len(ai_message.tool_calls) == 2

    # Check that we have a tool message from the regular tool
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1

    # The regular tool should have been executed
    regular_tool_message = next((m for m in tool_messages if m.name == "regular_tool"), None)
    assert regular_tool_message is not None
    assert "Regular tool result for: test query" in regular_tool_message.content

    # Verify that the structured response is available in the result
    assert "structured_response" in result
    assert result["structured_response"] is not None
    assert hasattr(result["structured_response"], "temperature")
    assert result["structured_response"].temperature == 72.0
    assert result["structured_response"].condition == "sunny"


def test_public_private_state_for_custom_middleware() -> None:
    """Test public and private state for custom middleware."""

    class CustomState(AgentState):
        omit_input: Annotated[str, OmitFromInput]
        omit_output: Annotated[str, OmitFromOutput]
        private_state: Annotated[str, PrivateStateAttr]

    class CustomMiddleware(AgentMiddleware[CustomState]):
        state_schema: type[CustomState] = CustomState

        def before_model(self, state: CustomState) -> dict[str, Any]:
            assert "omit_input" not in state
            assert "omit_output" in state
            assert "private_state" not in state
            return {"omit_input": "test", "omit_output": "test", "private_state": "test"}

    agent = create_agent(model=FakeToolCallingModel(), middleware=[CustomMiddleware()])
    result = agent.invoke(
        {
            "messages": [HumanMessage("Hello")],
            "omit_input": "test in",
            "private_state": "test in",
            "omit_output": "test in",
        }
    )
    assert "omit_input" in result
    assert "omit_output" not in result
    assert "private_state" not in result


def test_runtime_injected_into_middleware() -> None:
    """Test that the runtime is injected into the middleware."""

    class CustomMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime: Runtime) -> None:
            assert runtime is not None
            return None

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            assert request.runtime is not None
            return handler(request)

        def after_model(self, state: AgentState, runtime: Runtime) -> None:
            assert runtime is not None
            return None

    middleware = CustomMiddleware()

    agent = create_agent(model=FakeToolCallingModel(), middleware=[CustomMiddleware()])
    agent.invoke({"messages": [HumanMessage("Hello")]})


def test_injected_state_in_middleware_agent() -> None:
    """Test that custom state is properly injected into tools when using middleware."""

    class TestState(AgentState):
        test_state: str

    @tool(description="Test the state")
    def test_state(
        state: Annotated[TestState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> str:
        """Test tool that accesses injected state."""
        assert "test_state" in state
        return "success"

    class TestMiddleware(AgentMiddleware):
        state_schema = TestState

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {}, "id": "test_call_1", "name": "test_state"}],
                [],
            ]
        ),
        tools=[test_state],
        system_prompt="You are a helpful assistant.",
        middleware=[TestMiddleware()],
    )

    result = agent.invoke(
        {"test_state": "I love pizza", "messages": [HumanMessage("Call the test state tool")]}
    )

    messages = result["messages"]
    assert len(messages) == 4  # Human message, AI message with tool call, tool message, AI message

    # Find the tool message
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 1

    tool_message = tool_messages[0]
    assert tool_message.name == "test_state"
    assert "success" in tool_message.content
    assert tool_message.tool_call_id == "test_call_1"


def test_jump_to_is_ephemeral() -> None:
    class MyMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState) -> dict[str, Any]:
            assert "jump_to" not in state
            return {"jump_to": "model"}

        def after_model(self, state: AgentState) -> dict[str, Any]:
            assert "jump_to" not in state
            return {"jump_to": "model"}

    agent = create_agent(model=FakeToolCallingModel(), middleware=[MyMiddleware()])
    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert "jump_to" not in result


# Tests for PlanningMiddleware
def test_planning_middleware_initialization() -> None:
    """Test that PlanningMiddleware initializes correctly."""
    middleware = PlanningMiddleware()
    assert middleware.state_schema == PlanningState
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "write_todos"


@pytest.mark.parametrize(
    "original_prompt,expected_prompt_prefix",
    [
        ("Original prompt", "Original prompt\n\n## `write_todos`"),
        (None, "## `write_todos`"),
    ],
)
def test_planning_middleware_on_model_call(original_prompt, expected_prompt_prefix) -> None:
    """Test that wrap_model_call handles system prompts correctly."""
    middleware = PlanningMiddleware()
    model = FakeToolCallingModel()

    state: PlanningState = {"messages": [HumanMessage(content="Hello")]}

    request = ModelRequest(
        model=model,
        system_prompt=original_prompt,
        messages=[HumanMessage(content="Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state=state,
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call wrap_model_call to trigger the middleware logic
    middleware.wrap_model_call(request, mock_handler)
    # Check that the request was modified in place
    assert request.system_prompt.startswith(expected_prompt_prefix)


@pytest.mark.parametrize(
    "todos,expected_message",
    [
        ([], "Updated todo list to []"),
        (
            [{"content": "Task 1", "status": "pending"}],
            "Updated todo list to [{'content': 'Task 1', 'status': 'pending'}]",
        ),
        (
            [
                {"content": "Task 1", "status": "pending"},
                {"content": "Task 2", "status": "in_progress"},
            ],
            "Updated todo list to [{'content': 'Task 1', 'status': 'pending'}, {'content': 'Task 2', 'status': 'in_progress'}]",
        ),
        (
            [
                {"content": "Task 1", "status": "pending"},
                {"content": "Task 2", "status": "in_progress"},
                {"content": "Task 3", "status": "completed"},
            ],
            "Updated todo list to [{'content': 'Task 1', 'status': 'pending'}, {'content': 'Task 2', 'status': 'in_progress'}, {'content': 'Task 3', 'status': 'completed'}]",
        ),
    ],
)
def test_planning_middleware_write_todos_tool_execution(todos, expected_message) -> None:
    """Test that the write_todos tool executes correctly."""
    tool_call = {
        "args": {"todos": todos},
        "name": "write_todos",
        "type": "tool_call",
        "id": "test_call",
    }
    result = write_todos.invoke(tool_call)
    assert result.update["todos"] == todos
    assert result.update["messages"][0].content == expected_message


@pytest.mark.parametrize(
    "invalid_todos",
    [
        [{"content": "Task 1", "status": "invalid_status"}],
        [{"status": "pending"}],
    ],
)
def test_planning_middleware_write_todos_tool_validation_errors(invalid_todos) -> None:
    """Test that the write_todos tool rejects invalid input."""
    tool_call = {
        "args": {"todos": invalid_todos},
        "name": "write_todos",
        "type": "tool_call",
        "id": "test_call",
    }
    with pytest.raises(Exception):
        write_todos.invoke(tool_call)


def test_planning_middleware_agent_creation_with_middleware() -> None:
    """Test that an agent can be created with the planning middleware."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                    "name": "write_todos",
                    "type": "tool_call",
                    "id": "test_call",
                }
            ],
            [
                {
                    "args": {"todos": [{"content": "Task 1", "status": "in_progress"}]},
                    "name": "write_todos",
                    "type": "tool_call",
                    "id": "test_call",
                }
            ],
            [
                {
                    "args": {"todos": [{"content": "Task 1", "status": "completed"}]},
                    "name": "write_todos",
                    "type": "tool_call",
                    "id": "test_call",
                }
            ],
            [],
        ]
    )
    middleware = PlanningMiddleware()
    agent = create_agent(model=model, middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert result["todos"] == [{"content": "Task 1", "status": "completed"}]

    # human message (1)
    # ai message (2) - initial todo
    # tool message (3)
    # ai message (4) - updated todo
    # tool message (5)
    # ai message (6) - complete todo
    # tool message (7)
    # ai message (8) - no tool calls
    assert len(result["messages"]) == 8


def test_planning_middleware_custom_system_prompt() -> None:
    """Test that PlanningMiddleware can be initialized with custom system prompt."""
    custom_system_prompt = "Custom todo system prompt for testing"
    middleware = PlanningMiddleware(system_prompt=custom_system_prompt)
    model = FakeToolCallingModel()

    request = ModelRequest(
        model=model,
        system_prompt="Original prompt",
        messages=[HumanMessage(content="Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        model_settings={},
    )

    state: PlanningState = {"messages": [HumanMessage(content="Hello")]}

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call wrap_model_call to trigger the middleware logic
    middleware.wrap_model_call(request, mock_handler)
    # Check that the request was modified in place
    assert request.system_prompt == f"Original prompt\n\n{custom_system_prompt}"


def test_planning_middleware_custom_tool_description() -> None:
    """Test that PlanningMiddleware can be initialized with custom tool description."""
    custom_tool_description = "Custom tool description for testing"
    middleware = PlanningMiddleware(tool_description=custom_tool_description)

    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.description == custom_tool_description


def test_planning_middleware_custom_system_prompt_and_tool_description() -> None:
    """Test that PlanningMiddleware can be initialized with both custom prompts."""
    custom_system_prompt = "Custom system prompt"
    custom_tool_description = "Custom tool description"
    middleware = PlanningMiddleware(
        system_prompt=custom_system_prompt,
        tool_description=custom_tool_description,
    )

    # Verify system prompt
    model = FakeToolCallingModel()
    state: PlanningState = {"messages": [HumanMessage(content="Hello")]}

    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=[HumanMessage(content="Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state=state,
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call wrap_model_call to trigger the middleware logic
    middleware.wrap_model_call(request, mock_handler)
    # Check that the request was modified in place
    assert request.system_prompt == custom_system_prompt

    # Verify tool description
    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.description == custom_tool_description


def test_planning_middleware_default_prompts() -> None:
    """Test that PlanningMiddleware uses default prompts when none provided."""
    middleware = PlanningMiddleware()

    # Verify default system prompt
    assert middleware.system_prompt == WRITE_TODOS_SYSTEM_PROMPT

    # Verify default tool description
    assert middleware.tool_description == WRITE_TODOS_TOOL_DESCRIPTION
    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.description == WRITE_TODOS_TOOL_DESCRIPTION


def test_planning_middleware_custom_system_prompt() -> None:
    """Test that custom tool executes correctly in an agent."""
    middleware = PlanningMiddleware(system_prompt="call the write_todos tool")

    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "args": {"todos": [{"content": "Custom task", "status": "pending"}]},
                    "name": "write_todos",
                    "type": "tool_call",
                    "id": "test_call",
                }
            ],
            [],
        ]
    )

    agent = create_agent(model=model, middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert result["todos"] == [{"content": "Custom task", "status": "pending"}]
    # assert custom system prompt is in the first AI message
    assert "call the write_todos tool" in result["messages"][1].content


@tool
def simple_tool(input: str) -> str:
    """A simple tool"""
    return input


def test_middleware_unit_functionality():
    """Test that the middleware works as expected in isolation."""
    # Test with end behavior
    middleware = ModelCallLimitMiddleware(thread_limit=2, run_limit=1)

    # Mock runtime (not used in current implementation)
    runtime = None

    # Test when limits are not exceeded
    state = {"thread_model_call_count": 0, "run_model_call_count": 0}
    result = middleware.before_model(state, runtime)
    assert result is None

    # Test when thread limit is exceeded
    state = {"thread_model_call_count": 2, "run_model_call_count": 0}
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "thread limit (2/2)" in result["messages"][0].content

    # Test when run limit is exceeded
    state = {"thread_model_call_count": 1, "run_model_call_count": 1}
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "run limit (1/1)" in result["messages"][0].content

    # Test with error behavior
    middleware_exception = ModelCallLimitMiddleware(
        thread_limit=2, run_limit=1, exit_behavior="error"
    )

    # Test exception when thread limit exceeded
    state = {"thread_model_call_count": 2, "run_model_call_count": 0}
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware_exception.before_model(state, runtime)

    assert "thread limit (2/2)" in str(exc_info.value)

    # Test exception when run limit exceeded
    state = {"thread_model_call_count": 1, "run_model_call_count": 1}
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware_exception.before_model(state, runtime)

    assert "run limit (1/1)" in str(exc_info.value)


def test_thread_limit_with_create_agent():
    """Test that thread limits work correctly with create_agent."""
    model = FakeToolCallingModel()

    # Set thread limit to 1 (should be exceeded after 1 call)
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[ModelCallLimitMiddleware(thread_limit=1)],
        checkpointer=InMemorySaver(),
    )

    # First invocation should work - 1 model call, within thread limit
    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]}, {"configurable": {"thread_id": "thread1"}}
    )

    # Should complete successfully with 1 model call
    assert "messages" in result
    assert len(result["messages"]) == 2  # Human + AI messages

    # Second invocation in same thread should hit thread limit
    # The agent should jump to end after detecting the limit
    result2 = agent.invoke(
        {"messages": [HumanMessage("Hello again")]}, {"configurable": {"thread_id": "thread1"}}
    )

    assert "messages" in result2
    # The agent should have detected the limit and jumped to end with a limit exceeded message
    # So we should have: previous messages + new human message + limit exceeded AI message
    assert len(result2["messages"]) == 4  # Previous Human + AI + New Human + Limit AI
    assert isinstance(result2["messages"][0], HumanMessage)  # First human
    assert isinstance(result2["messages"][1], AIMessage)  # First AI response
    assert isinstance(result2["messages"][2], HumanMessage)  # Second human
    assert isinstance(result2["messages"][3], AIMessage)  # Limit exceeded message
    assert "thread limit" in result2["messages"][3].content


def test_run_limit_with_create_agent():
    """Test that run limits work correctly with create_agent."""
    # Create a model that will make 2 calls
    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "simple_tool", "args": {"input": "test"}, "id": "1"}],
            [],  # No tool calls on second call
        ]
    )

    # Set run limit to 1 (should be exceeded after 1 call)
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[ModelCallLimitMiddleware(run_limit=1)],
        checkpointer=InMemorySaver(),
    )

    # This should hit the run limit after the first model call
    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]}, {"configurable": {"thread_id": "thread1"}}
    )

    assert "messages" in result
    # The agent should have made 1 model call then jumped to end with limit exceeded message
    # So we should have: Human + AI + Tool + Limit exceeded AI message
    assert len(result["messages"]) == 4  # Human + AI + Tool + Limit AI
    assert isinstance(result["messages"][0], HumanMessage)
    assert isinstance(result["messages"][1], AIMessage)
    assert isinstance(result["messages"][2], ToolMessage)
    assert isinstance(result["messages"][3], AIMessage)  # Limit exceeded message
    assert "run limit" in result["messages"][3].content


def test_middleware_initialization_validation():
    """Test that middleware initialization validates parameters correctly."""
    # Test that at least one limit must be specified
    with pytest.raises(ValueError, match="At least one limit must be specified"):
        ModelCallLimitMiddleware()

    # Test invalid exit behavior
    with pytest.raises(ValueError, match="Invalid exit_behavior"):
        ModelCallLimitMiddleware(thread_limit=5, exit_behavior="invalid")

    # Test valid initialization
    middleware = ModelCallLimitMiddleware(thread_limit=5, run_limit=3)
    assert middleware.thread_limit == 5
    assert middleware.run_limit == 3
    assert middleware.exit_behavior == "end"

    # Test with only thread limit
    middleware = ModelCallLimitMiddleware(thread_limit=5)
    assert middleware.thread_limit == 5
    assert middleware.run_limit is None

    # Test with only run limit
    middleware = ModelCallLimitMiddleware(run_limit=3)
    assert middleware.thread_limit is None
    assert middleware.run_limit == 3


def test_exception_error_message():
    """Test that the exception provides clear error messages."""
    middleware = ModelCallLimitMiddleware(thread_limit=2, run_limit=1, exit_behavior="error")

    # Test thread limit exceeded
    state = {"thread_model_call_count": 2, "run_model_call_count": 0}
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware.before_model(state, None)

    error_msg = str(exc_info.value)
    assert "Model call limits exceeded" in error_msg
    assert "thread limit (2/2)" in error_msg

    # Test run limit exceeded
    state = {"thread_model_call_count": 0, "run_model_call_count": 1}
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware.before_model(state, None)

    error_msg = str(exc_info.value)
    assert "Model call limits exceeded" in error_msg
    assert "run limit (1/1)" in error_msg

    # Test both limits exceeded
    state = {"thread_model_call_count": 2, "run_model_call_count": 1}
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware.before_model(state, None)

    error_msg = str(exc_info.value)
    assert "Model call limits exceeded" in error_msg
    assert "thread limit (2/2)" in error_msg
    assert "run limit (1/1)" in error_msg


def test_run_limit_resets_between_invocations() -> None:
    """Test that run_model_call_count resets between invocations, but thread_model_call_count accumulates."""

    # First: No tool calls per invocation, so model does not increment call counts internally
    middleware = ModelCallLimitMiddleware(thread_limit=3, run_limit=1, exit_behavior="error")
    model = FakeToolCallingModel(
        tool_calls=[[], [], [], []]
    )  # No tool calls, so only model call per run

    agent = create_agent(model=model, middleware=[middleware], checkpointer=InMemorySaver())

    thread_config = {"configurable": {"thread_id": "test_thread"}}
    agent.invoke({"messages": [HumanMessage("Hello")]}, thread_config)
    agent.invoke({"messages": [HumanMessage("Hello again")]}, thread_config)
    agent.invoke({"messages": [HumanMessage("Hello third")]}, thread_config)

    # Fourth run: should raise, thread_model_call_count == 3 (limit)
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        agent.invoke({"messages": [HumanMessage("Hello fourth")]}, thread_config)
    error_msg = str(exc_info.value)
    assert "thread limit (3/3)" in error_msg


# Async Middleware Tests
async def test_create_agent_async_invoke() -> None:
    """Test async invoke with async middleware hooks."""
    calls = []

    class AsyncMiddleware(AgentMiddleware):
        async def abefore_model(self, state, runtime) -> None:
            calls.append("AsyncMiddleware.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            calls.append("AsyncMiddleware.awrap_model_call")
            request.messages.append(HumanMessage("async middleware message"))
            return await handler(request)

        async def aafter_model(self, state, runtime) -> None:
            calls.append("AsyncMiddleware.aafter_model")

    @tool
    def my_tool(input: str) -> str:
        """A great tool"""
        calls.append("my_tool")
        return input.upper()

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"input": "yo"}, "id": "1", "name": "my_tool"}],
                [],
            ]
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[AsyncMiddleware()],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("hello")]})

    # Should have:
    # 1. Original hello message
    # 2. Async middleware message (first invoke)
    # 3. AI message with tool call
    # 4. Tool message
    # 5. Async middleware message (second invoke)
    # 6. Final AI message
    assert len(result["messages"]) == 6
    assert result["messages"][0].content == "hello"
    assert result["messages"][1].content == "async middleware message"
    assert calls == [
        "AsyncMiddleware.abefore_model",
        "AsyncMiddleware.awrap_model_call",
        "AsyncMiddleware.aafter_model",
        "my_tool",
        "AsyncMiddleware.abefore_model",
        "AsyncMiddleware.awrap_model_call",
        "AsyncMiddleware.aafter_model",
    ]


async def test_create_agent_async_invoke_multiple_middleware() -> None:
    """Test async invoke with multiple async middleware hooks."""
    calls = []

    class AsyncMiddlewareOne(AgentMiddleware):
        async def abefore_model(self, state, runtime) -> None:
            calls.append("AsyncMiddlewareOne.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            calls.append("AsyncMiddlewareOne.awrap_model_call")
            return await handler(request)

        async def aafter_model(self, state, runtime) -> None:
            calls.append("AsyncMiddlewareOne.aafter_model")

    class AsyncMiddlewareTwo(AgentMiddleware):
        async def abefore_model(self, state, runtime) -> None:
            calls.append("AsyncMiddlewareTwo.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            calls.append("AsyncMiddlewareTwo.awrap_model_call")
            return await handler(request)

        async def aafter_model(self, state, runtime) -> None:
            calls.append("AsyncMiddlewareTwo.aafter_model")

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[AsyncMiddlewareOne(), AsyncMiddlewareTwo()],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("hello")]})

    assert calls == [
        "AsyncMiddlewareOne.abefore_model",
        "AsyncMiddlewareTwo.abefore_model",
        "AsyncMiddlewareOne.awrap_model_call",
        "AsyncMiddlewareTwo.awrap_model_call",
        "AsyncMiddlewareTwo.aafter_model",
        "AsyncMiddlewareOne.aafter_model",
    ]


async def test_create_agent_async_jump() -> None:
    """Test async invoke with async middleware using jump_to."""
    calls = []

    class AsyncMiddlewareOne(AgentMiddleware):
        async def abefore_model(self, state, runtime) -> None:
            calls.append("AsyncMiddlewareOne.abefore_model")

    class AsyncMiddlewareTwo(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        async def abefore_model(self, state, runtime) -> dict[str, Any]:
            calls.append("AsyncMiddlewareTwo.abefore_model")
            return {"jump_to": "end"}

    @tool
    def my_tool(input: str) -> str:
        """A great tool"""
        calls.append("my_tool")
        return input.upper()

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[ToolCall(id="1", name="my_tool", args={"input": "yo"})]],
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[AsyncMiddlewareOne(), AsyncMiddlewareTwo()],
    )

    result = await agent.ainvoke({"messages": []})

    assert result == {"messages": []}
    assert calls == ["AsyncMiddlewareOne.abefore_model", "AsyncMiddlewareTwo.abefore_model"]


async def test_create_agent_mixed_sync_async_middleware() -> None:
    """Test async invoke with mixed sync and async middleware."""
    calls = []

    class SyncMiddleware(AgentMiddleware):
        def before_model(self, state, runtime) -> None:
            calls.append("SyncMiddleware.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("SyncMiddleware.wrap_model_call")
            return handler(request)

        def after_model(self, state, runtime) -> None:
            calls.append("SyncMiddleware.after_model")

    class AsyncMiddleware(AgentMiddleware):
        async def abefore_model(self, state, runtime) -> None:
            calls.append("AsyncMiddleware.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            calls.append("AsyncMiddleware.awrap_model_call")
            return await handler(request)

        async def aafter_model(self, state, runtime) -> None:
            calls.append("AsyncMiddleware.aafter_model")

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[SyncMiddleware(), AsyncMiddleware()],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("hello")]})

    # In async mode, both sync and async middleware should work
    # Note: Sync wrap_model_call is not called when running in async mode,
    # as the async version is preferred
    assert calls == [
        "SyncMiddleware.before_model",
        "AsyncMiddleware.abefore_model",
        "AsyncMiddleware.awrap_model_call",
        "AsyncMiddleware.aafter_model",
        "SyncMiddleware.after_model",
    ]


# Tests for wrap_model_call hook
def test_wrap_model_call_hook() -> None:
    """Test that wrap_model_call hook is called on model errors."""
    call_count = {"value": 0}

    class FailingModel(BaseChatModel):
        """Model that fails on first call, succeeds on second."""

        def _generate(self, messages, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise ValueError("First call fails")
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Success on retry"))]
            )

        @property
        def _llm_type(self):
            return "failing"

    class RetryMiddleware(AgentMiddleware):
        def __init__(self):
            super().__init__()
            self.retry_count = 0

        def wrap_model_call(self, request, handler):
            try:
                return handler(request)
            except Exception:
                # Retry on error
                self.retry_count += 1
                return handler(request)

    failing_model = FailingModel()
    retry_middleware = RetryMiddleware()

    agent = create_agent(model=failing_model, middleware=[retry_middleware])

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Should have retried once
    assert retry_middleware.retry_count == 1
    # Should have succeeded on second attempt
    assert len(result["messages"]) == 2
    assert result["messages"][1].content == "Success on retry"


def test_wrap_model_call_retry_count() -> None:
    """Test that wrap_model_call can retry multiple times."""

    class AlwaysFailingModel(BaseChatModel):
        """Model that always fails."""

        def _generate(self, messages, **kwargs):
            raise ValueError("Always fails")

        @property
        def _llm_type(self):
            return "always_failing"

    class AttemptTrackingMiddleware(AgentMiddleware):
        def __init__(self):
            super().__init__()
            self.attempts = []

        def wrap_model_call(self, request, handler):
            max_retries = 3
            last_exception = None
            for attempt in range(max_retries):
                self.attempts.append(attempt + 1)
                try:
                    return handler(request)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        continue  # Retry

            # All retries failed, re-raise the last exception
            if last_exception:
                raise last_exception

    model = AlwaysFailingModel()
    tracker = AttemptTrackingMiddleware()

    agent = create_agent(model=model, middleware=[tracker])

    with pytest.raises(ValueError, match="Always fails"):
        agent.invoke({"messages": [HumanMessage("Test")]})

    # Should have attempted 3 times
    assert tracker.attempts == [1, 2, 3]


def test_wrap_model_call_no_retry() -> None:
    """Test that error is propagated when middleware doesn't retry."""

    class FailingModel(BaseChatModel):
        """Model that always fails."""

        def _generate(self, messages, **kwargs):
            raise ValueError("Model error")

        @property
        def _llm_type(self):
            return "failing"

    class NoRetryMiddleware(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return handler(request)

    agent = create_agent(model=FailingModel(), middleware=[NoRetryMiddleware()])

    with pytest.raises(ValueError, match="Model error"):
        agent.invoke({"messages": [HumanMessage("Test")]})


def test_model_fallback_middleware() -> None:
    """Test ModelFallbackMiddleware with fallback models only."""

    class FailingModel(BaseChatModel):
        """Model that always fails."""

        def _generate(self, messages, **kwargs):
            raise ValueError("Primary model failed")

        @property
        def _llm_type(self):
            return "failing"

    class SuccessModel(BaseChatModel):
        """Model that succeeds."""

        def _generate(self, messages, **kwargs):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Fallback success"))]
            )

        @property
        def _llm_type(self):
            return "success"

    primary = FailingModel()
    fallback = SuccessModel()

    # Only pass fallback models to middleware (not the primary)
    fallback_middleware = ModelFallbackMiddleware(fallback)

    agent = create_agent(model=primary, middleware=[fallback_middleware])

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Should have succeeded with fallback model
    assert len(result["messages"]) == 2
    assert result["messages"][1].content == "Fallback success"


def test_model_fallback_middleware_exhausted() -> None:
    """Test ModelFallbackMiddleware when all models fail."""

    class AlwaysFailingModel(BaseChatModel):
        """Model that always fails."""

        def __init__(self, name: str):
            super().__init__()
            self.name = name

        def _generate(self, messages, **kwargs):
            raise ValueError(f"{self.name} failed")

        @property
        def _llm_type(self):
            return self.name

    primary = AlwaysFailingModel("primary")
    fallback1 = AlwaysFailingModel("fallback1")
    fallback2 = AlwaysFailingModel("fallback2")

    # Primary fails (attempt 1), then fallback1 (attempt 2), then fallback2 (attempt 3)
    fallback_middleware = ModelFallbackMiddleware(fallback1, fallback2)

    agent = create_agent(model=primary, middleware=[fallback_middleware])

    # Should fail with the last fallback's error
    with pytest.raises(ValueError, match="fallback2 failed"):
        agent.invoke({"messages": [HumanMessage("Test")]})


def test_model_fallback_middleware_initialization() -> None:
    """Test ModelFallbackMiddleware initialization."""

    # Test with no models - now a TypeError (missing required argument)
    with pytest.raises(TypeError):
        ModelFallbackMiddleware()  # type: ignore[call-arg]

    # Test with one fallback model (valid)
    middleware = ModelFallbackMiddleware(FakeToolCallingModel())
    assert len(middleware.models) == 1

    # Test with multiple fallback models
    middleware = ModelFallbackMiddleware(FakeToolCallingModel(), FakeToolCallingModel())
    assert len(middleware.models) == 2


def test_wrap_model_call_max_attempts() -> None:
    """Test that middleware controls termination via retry limits."""

    class AlwaysFailingModel(BaseChatModel):
        """Model that always fails."""

        def _generate(self, messages, **kwargs):
            raise ValueError("Always fails")

        @property
        def _llm_type(self):
            return "always_failing"

    class LimitedRetryMiddleware(AgentMiddleware):
        """Middleware that limits its own retries."""

        def __init__(self, max_retries: int = 10):
            super().__init__()
            self.max_retries = max_retries
            self.attempt_count = 0

        def wrap_model_call(self, request, handler):
            last_exception = None
            for attempt in range(self.max_retries):
                self.attempt_count += 1
                try:
                    return handler(request)
                except Exception as e:
                    last_exception = e
                    # Continue to retry

            # All retries exhausted, re-raise the last error
            if last_exception:
                raise last_exception

    model = AlwaysFailingModel()
    middleware = LimitedRetryMiddleware(max_retries=10)

    agent = create_agent(model=model, middleware=[middleware])

    # Should fail with the model's error after middleware stops retrying
    with pytest.raises(ValueError, match="Always fails"):
        agent.invoke({"messages": [HumanMessage("Test")]})

    # Should have attempted exactly 10 times as configured
    assert middleware.attempt_count == 10


async def test_wrap_model_call_async() -> None:
    """Test wrap_model_call hook with async model execution."""
    call_count = {"value": 0}

    class AsyncFailingModel(BaseChatModel):
        """Model that fails on first async call, succeeds on second."""

        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="sync"))])

        async def _agenerate(self, messages, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise ValueError("First async call fails")
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Async retry success"))]
            )

        @property
        def _llm_type(self):
            return "async_failing"

    class AsyncRetryMiddleware(AgentMiddleware):
        def __init__(self):
            super().__init__()
            self.retry_count = 0

        async def awrap_model_call(self, request, handler):
            try:
                return await handler(request)
            except Exception:
                # Retry on error
                self.retry_count += 1
                return await handler(request)

    failing_model = AsyncFailingModel()
    retry_middleware = AsyncRetryMiddleware()

    agent = create_agent(model=failing_model, middleware=[retry_middleware])

    result = await agent.ainvoke({"messages": [HumanMessage("Test")]})

    # Should have retried once
    assert retry_middleware.retry_count == 1
    # Should have succeeded on second attempt
    assert len(result["messages"]) == 2
    assert result["messages"][1].content == "Async retry success"


def test_wrap_model_call_rewrite_response() -> None:
    """Test that middleware can rewrite model responses."""

    class SimpleModel(BaseChatModel):
        """Model that returns a simple response."""

        def _generate(self, messages, **kwargs):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Original response"))]
            )

        @property
        def _llm_type(self):
            return "simple"

    class ResponseRewriteMiddleware(AgentMiddleware):
        """Middleware that rewrites the response."""

        def wrap_model_call(self, request, handler):
            result = handler(request)

            # Rewrite the response
            return AIMessage(content=f"REWRITTEN: {result.content}")

    model = SimpleModel()
    middleware = ResponseRewriteMiddleware()

    agent = create_agent(model=model, middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Response should be rewritten by middleware
    assert result["messages"][1].content == "REWRITTEN: Original response"


def test_wrap_model_call_convert_error_to_response() -> None:
    """Test that middleware can convert errors to successful responses."""

    class AlwaysFailingModel(BaseChatModel):
        """Model that always fails."""

        def _generate(self, messages, **kwargs):
            raise ValueError("Model error")

        @property
        def _llm_type(self):
            return "failing"

    class ErrorToResponseMiddleware(AgentMiddleware):
        """Middleware that converts errors to success responses."""

        def wrap_model_call(self, request, handler):
            try:
                return handler(request)
            except Exception as e:
                # Convert error to success response
                return AIMessage(content=f"Error occurred: {e}. Using fallback response.")

    model = AlwaysFailingModel()
    middleware = ErrorToResponseMiddleware()

    agent = create_agent(model=model, middleware=[middleware])

    # Should not raise, middleware converts error to response
    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Response should be the fallback from middleware
    assert "Error occurred" in result["messages"][1].content
    assert "fallback response" in result["messages"][1].content


def test_create_agent_sync_invoke_with_only_async_middleware_raises_error() -> None:
    """Test that sync invoke with only async middleware works via run_in_executor."""

    class AsyncOnlyMiddleware(AgentMiddleware):
        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            return await handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[AsyncOnlyMiddleware()],
    )

    # This should work now via run_in_executor
    result = agent.invoke({"messages": [HumanMessage("hello")]})
    assert result is not None
    assert "messages" in result


def test_create_agent_sync_invoke_with_mixed_middleware() -> None:
    """Test that sync invoke works with mixed sync/async middleware when sync versions exist."""
    calls = []

    class MixedMiddleware(AgentMiddleware):
        def before_model(self, state, runtime) -> None:
            calls.append("MixedMiddleware.before_model")

        async def abefore_model(self, state, runtime) -> None:
            calls.append("MixedMiddleware.abefore_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("MixedMiddleware.wrap_model_call")
            return handler(request)

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            calls.append("MixedMiddleware.awrap_model_call")
            return await handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[MixedMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("hello")]})

    # In sync mode, only sync methods should be called
    assert calls == [
        "MixedMiddleware.before_model",
        "MixedMiddleware.wrap_model_call",
    ]
