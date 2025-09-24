import pytest
from typing import Any
from unittest.mock import patch
from types import ModuleType

from syrupy.assertion import SnapshotAssertion

import warnings
from langgraph.runtime import Runtime
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import tool, InjectedToolCallId

from langchain.agents.middleware_agent import create_agent
from langchain.agents.tool_node import InjectedState
from langchain.agents.middleware.human_in_the_loop import (
    HumanInTheLoopMiddleware,
    ActionRequest,
)
from langchain.agents.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    AgentState,
    OmitFromInput,
    OmitFromOutput,
    PrivateStateAttr,
)

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from langchain.agents.structured_output import ToolStrategy

from .messages import _AnyIdHumanMessage, _AnyIdToolMessage
from .model import FakeToolCallingModel


def test_create_agent_diagram(
    snapshot: SnapshotAssertion,
):
    class NoopOne(AgentMiddleware):
        def before_model(self, state):
            pass

    class NoopTwo(AgentMiddleware):
        def before_model(self, state):
            pass

    class NoopThree(AgentMiddleware):
        def before_model(self, state):
            pass

    class NoopFour(AgentMiddleware):
        def after_model(self, state):
            pass

    class NoopFive(AgentMiddleware):
        def after_model(self, state):
            pass

    class NoopSix(AgentMiddleware):
        def after_model(self, state):
            pass

    class NoopSeven(AgentMiddleware):
        def before_model(self, state):
            pass

        def after_model(self, state):
            pass

    class NoopEight(AgentMiddleware):
        def before_model(self, state):
            pass

        def after_model(self, state):
            pass

    class NoopNine(AgentMiddleware):
        def before_model(self, state):
            pass

        def after_model(self, state):
            pass

    class NoopTen(AgentMiddleware):
        def before_model(self, state):
            pass

        def modify_model_request(self, request, state):
            pass

        def after_model(self, state):
            pass

    class NoopEleven(AgentMiddleware):
        def before_model(self, state):
            pass

        def modify_model_request(self, request, state):
            pass

        def after_model(self, state):
            pass

    agent_zero = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
    )

    assert agent_zero.compile().get_graph().draw_mermaid() == snapshot

    agent_one = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne()],
    )

    assert agent_one.compile().get_graph().draw_mermaid() == snapshot

    agent_two = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne(), NoopTwo()],
    )

    assert agent_two.compile().get_graph().draw_mermaid() == snapshot

    agent_three = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne(), NoopTwo(), NoopThree()],
    )

    assert agent_three.compile().get_graph().draw_mermaid() == snapshot

    agent_four = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour()],
    )

    assert agent_four.compile().get_graph().draw_mermaid() == snapshot

    agent_five = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour(), NoopFive()],
    )

    assert agent_five.compile().get_graph().draw_mermaid() == snapshot

    agent_six = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour(), NoopFive(), NoopSix()],
    )

    assert agent_six.compile().get_graph().draw_mermaid() == snapshot

    agent_seven = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven()],
    )

    assert agent_seven.compile().get_graph().draw_mermaid() == snapshot

    agent_eight = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight()],
    )

    assert agent_eight.compile().get_graph().draw_mermaid() == snapshot

    agent_nine = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight(), NoopNine()],
    )

    assert agent_nine.compile().get_graph().draw_mermaid() == snapshot

    agent_ten = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopTen()],
    )

    assert agent_ten.compile().get_graph().draw_mermaid() == snapshot

    agent_eleven = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopTen(), NoopEleven()],
    )

    assert agent_eleven.compile().get_graph().draw_mermaid() == snapshot


def test_create_agent_invoke(
    snapshot: SnapshotAssertion,
    sync_checkpointer: BaseCheckpointSaver,
):
    calls = []

    class NoopSeven(AgentMiddleware):
        def before_model(self, state):
            calls.append("NoopSeven.before_model")

        def modify_model_request(self, request, state):
            calls.append("NoopSeven.modify_model_request")
            return request

        def after_model(self, state):
            calls.append("NoopSeven.after_model")

    class NoopEight(AgentMiddleware):
        def before_model(self, state):
            calls.append("NoopEight.before_model")

        def modify_model_request(self, request, state):
            calls.append("NoopEight.modify_model_request")
            return request

        def after_model(self, state):
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
    ).compile(checkpointer=sync_checkpointer)

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
        "NoopSeven.modify_model_request",
        "NoopEight.modify_model_request",
        "NoopEight.after_model",
        "NoopSeven.after_model",
        "my_tool",
        "NoopSeven.before_model",
        "NoopEight.before_model",
        "NoopSeven.modify_model_request",
        "NoopEight.modify_model_request",
        "NoopEight.after_model",
        "NoopSeven.after_model",
    ]


def test_create_agent_jump(
    snapshot: SnapshotAssertion,
    sync_checkpointer: BaseCheckpointSaver,
):
    calls = []

    class NoopSeven(AgentMiddleware):
        def before_model(self, state):
            calls.append("NoopSeven.before_model")

        def modify_model_request(self, request, state):
            calls.append("NoopSeven.modify_model_request")
            return request

        def after_model(self, state):
            calls.append("NoopSeven.after_model")

    class NoopEight(AgentMiddleware):
        before_model_jump_to = [END]

        def before_model(self, state) -> dict[str, Any]:
            calls.append("NoopEight.before_model")
            return {"jump_to": END}

        def modify_model_request(self, request, state) -> ModelRequest:
            calls.append("NoopEight.modify_model_request")
            return request

        def after_model(self, state):
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
    ).compile(checkpointer=sync_checkpointer)

    if isinstance(sync_checkpointer, InMemorySaver):
        assert agent_one.get_graph().draw_mermaid() == snapshot

    thread1 = {"configurable": {"thread_id": "1"}}
    assert agent_one.invoke({"messages": []}, thread1) == {"messages": []}
    assert calls == ["NoopSeven.before_model", "NoopEight.before_model"]


# Tests for HumanInTheLoopMiddleware
def test_human_in_the_loop_middleware_initialization() -> None:
    """Test HumanInTheLoopMiddleware initialization."""

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {"allow_accept": True, "allow_edit": True, "allow_respond": True}
        },
        description_prefix="Custom prefix",
    )

    assert middleware.interrupt_on == {
        "test_tool": {"allow_accept": True, "allow_edit": True, "allow_respond": True}
    }
    assert middleware.description_prefix == "Custom prefix"


def test_human_in_the_loop_middleware_no_interrupts_needed() -> None:
    """Test HumanInTheLoopMiddleware when no interrupts are needed."""

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {"allow_respond": True, "allow_edit": True, "allow_accept": True}
        }
    )

    # Test with no messages
    state: dict[str, Any] = {"messages": []}
    result = middleware.after_model(state)
    assert result is None

    # Test with message but no tool calls
    state = {"messages": [HumanMessage(content="Hello"), AIMessage(content="Hi there")]}
    result = middleware.after_model(state)
    assert result is None

    # Test with tool calls that don't require interrupts
    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "other_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}
    result = middleware.after_model(state)
    assert result is None


def test_human_in_the_loop_middleware_single_tool_accept() -> None:
    """Test HumanInTheLoopMiddleware with single tool accept response."""

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {"allow_respond": True, "allow_edit": True, "allow_accept": True}
        }
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_accept(requests):
        return [{"type": "accept", "args": None}]

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_accept):
        result = middleware.after_model(state)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0] == ai_message
        assert result["messages"][0].tool_calls == ai_message.tool_calls

    state["messages"].append(
        ToolMessage(content="Tool message", name="test_tool", tool_call_id="1")
    )
    state["messages"].append(AIMessage(content="test_tool called with result: Tool message"))

    result = middleware.after_model(state)
    # No interrupts needed
    assert result is None


def test_human_in_the_loop_middleware_single_tool_edit() -> None:
    """Test HumanInTheLoopMiddleware with single tool edit response."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {"allow_respond": True, "allow_edit": True, "allow_accept": True}
        }
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_edit(requests):
        return [
            {
                "type": "edit",
                "args": ActionRequest(
                    action="test_tool",
                    args={"input": "edited"},
                ),
            }
        ]

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_edit):
        result = middleware.after_model(state)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls[0]["args"] == {"input": "edited"}
        assert result["messages"][0].tool_calls[0]["id"] == "1"  # ID should be preserved


def test_human_in_the_loop_middleware_single_tool_response() -> None:
    """Test HumanInTheLoopMiddleware with single tool response with custom message."""

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {"allow_respond": True, "allow_edit": True, "allow_accept": True}
        }
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_response(requests):
        return [{"type": "response", "args": "Custom response message"}]

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_response
    ):
        result = middleware.after_model(state)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1  # Only tool message when no approved tool calls
        assert isinstance(result["messages"][0], ToolMessage)
        assert result["messages"][0].content == "Custom response message"
        assert result["messages"][0].name == "test_tool"
        assert result["messages"][0].tool_call_id == "1"


def test_human_in_the_loop_middleware_multiple_tools_mixed_responses() -> None:
    """Test HumanInTheLoopMiddleware with multiple tools and mixed response types."""

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "get_forecast": {"allow_accept": True, "allow_edit": True, "allow_respond": True},
            "get_temperature": {"allow_accept": True, "allow_edit": True, "allow_respond": True},
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
        return [
            {"type": "accept", "args": None},
            {"type": "response", "args": "User rejected this tool call"},
        ]

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_mixed_responses
    ):
        result = middleware.after_model(state)
        assert result is not None
        assert "messages" in result
        assert (
            len(result["messages"]) == 2
        )  # AI message with accepted tool call + tool message for rejected

        # First message should be the AI message with updated tool calls
        updated_ai_message = result["messages"][0]
        assert len(updated_ai_message.tool_calls) == 1  # Only accepted tool call
        assert updated_ai_message.tool_calls[0]["name"] == "get_forecast"  # Accepted

        # Second message should be the tool message for the rejected tool call
        tool_message = result["messages"][1]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.content == "User rejected this tool call"
        assert tool_message.name == "get_temperature"


def test_human_in_the_loop_middleware_multiple_tools_edit_responses() -> None:
    """Test HumanInTheLoopMiddleware with multiple tools and edit responses."""

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "get_forecast": {"allow_accept": True, "allow_edit": True, "allow_respond": True},
            "get_temperature": {"allow_accept": True, "allow_edit": True, "allow_respond": True},
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
        return [
            {
                "type": "edit",
                "args": ActionRequest(
                    action="get_forecast",
                    args={"location": "New York"},
                ),
            },
            {
                "type": "edit",
                "args": ActionRequest(
                    action="get_temperature",
                    args={"location": "New York"},
                ),
            },
        ]

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_edit_responses
    ):
        result = middleware.after_model(state)
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
        interrupt_on={
            "test_tool": {"allow_accept": True, "allow_edit": True, "allow_respond": True}
        }
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_edit_with_args(requests):
        return [
            {
                "type": "edit",
                "args": ActionRequest(
                    action="test_tool",
                    args={"input": "modified"},
                ),
            }
        ]

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        side_effect=mock_edit_with_args,
    ):
        result = middleware.after_model(state)
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
        interrupt_on={
            "test_tool": {"allow_accept": True, "allow_edit": True, "allow_respond": True}
        }
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_unknown(requests):
        return [{"type": "unknown", "args": None}]

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_unknown):
        with pytest.raises(
            ValueError,
            match=r"Unexpected human response: {'type': 'unknown', 'args': None}. Response action 'unknown' is not allowed for tool 'test_tool'. Expected one of \['accept', 'edit', 'response'\] based on the tool's configuration.",
        ):
            middleware.after_model(state)


def test_human_in_the_loop_middleware_disallowed_action() -> None:
    """Test HumanInTheLoopMiddleware with action not allowed by tool config."""

    # edit is not allowed by tool config
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "test_tool": {"allow_respond": True, "allow_edit": False, "allow_accept": True}
        }
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    def mock_disallowed_action(requests):
        return [
            {
                "type": "edit",
                "args": ActionRequest(
                    action="test_tool",
                    args={"input": "modified"},
                ),
            }
        ]

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        side_effect=mock_disallowed_action,
    ):
        with pytest.raises(
            ValueError,
            match=r"Unexpected human response: {'type': 'edit', 'args': {'action': 'test_tool', 'args': {'input': 'modified'}}}. Response action 'edit' is not allowed for tool 'test_tool'. Expected one of \['accept', 'response'\] based on the tool's configuration.",
        ):
            middleware.after_model(state)


def test_human_in_the_loop_middleware_mixed_auto_approved_and_interrupt() -> None:
    """Test HumanInTheLoopMiddleware with mix of auto-approved and interrupt tools."""

    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "interrupt_tool": {"allow_respond": True, "allow_edit": True, "allow_accept": True}
        }
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
        return [{"type": "accept", "args": None}]

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_accept):
        result = middleware.after_model(state)
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
        interrupt_on={
            "test_tool": {"allow_accept": True, "allow_edit": True, "allow_respond": True}
        },
        description_prefix="Custom prefix",
    )

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test", "location": "SF"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    captured_requests = []

    def mock_capture_requests(requests):
        captured_requests.extend(requests)
        return [{"type": "accept", "args": None}]

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_capture_requests
    ):
        middleware.after_model(state)

        assert len(captured_requests) == 1
        request = captured_requests[0]

        assert "action_request" in request
        assert "config" in request
        assert "description" in request

        assert request["action_request"]["action"] == "test_tool"
        assert request["action_request"]["args"] == {"input": "test", "location": "SF"}
        expected_config = {"allow_accept": True, "allow_edit": True, "allow_respond": True}
        assert request["config"] == expected_config
        assert "Custom prefix" in request["description"]
        assert "Tool: test_tool" in request["description"]
        assert "Args: {'input': 'test', 'location': 'SF'}" in request["description"]


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
        return_value=[{"type": "accept", "args": None}],
    ):
        result = middleware.after_model(state)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls == ai_message.tool_calls

    # Test edit
    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value=[
            {
                "type": "edit",
                "args": ActionRequest(
                    action="test_tool",
                    args={"input": "edited"},
                ),
            }
        ],
    ):
        result = middleware.after_model(state)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls[0]["args"] == {"input": "edited"}

    middleware = HumanInTheLoopMiddleware(interrupt_on={"test_tool": False})

    result = middleware.after_model(state)
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
        return_value=[],  # No responses for 1 tool call
    ):
        with pytest.raises(
            ValueError,
            match=r"Number of human responses \(0\) does not match number of hanging tool calls \(1\)\.",
        ):
            middleware.after_model(state)

    # Test with too many responses
    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt",
        return_value=[
            {"type": "accept", "args": None},
            {"type": "accept", "args": None},
        ],  # 2 responses for 1 tool call
    ):
        with pytest.raises(
            ValueError,
            match=r"Number of human responses \(2\) does not match number of hanging tool calls \(1\)\.",
        ):
            middleware.after_model(state)


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
        model_settings={},
    )

    assert middleware.modify_model_request(fake_request).model_settings == {
        "cache_control": {"type": "ephemeral", "ttl": "5m"}
    }


def test_anthropic_prompt_caching_middleware_unsupported_model() -> None:
    """Test AnthropicPromptCachingMiddleware with unsupported model."""
    fake_request = ModelRequest(
        model=FakeToolCallingModel(),
        messages=[HumanMessage("Hello")],
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        model_settings={},
    )

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="raise")

    with pytest.raises(
        ValueError,
        match="AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models. Please install langchain-anthropic.",
    ):
        middleware.modify_model_request(fake_request)

    langchain_anthropic = ModuleType("langchain_anthropic")

    class MockChatAnthropic:
        pass

    langchain_anthropic.ChatAnthropic = MockChatAnthropic

    with patch.dict("sys.modules", {"langchain_anthropic": langchain_anthropic}):
        with pytest.raises(
            ValueError,
            match="AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models, not instances of",
        ):
            middleware.modify_model_request(fake_request)

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="warn")

    with warnings.catch_warnings(record=True) as w:
        result = middleware.modify_model_request(fake_request)
        assert len(w) == 1
        assert (
            "AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models. Please install langchain-anthropic."
            in str(w[-1].message)
        )
        assert result == fake_request

    with warnings.catch_warnings(record=True) as w:
        with patch.dict("sys.modules", {"langchain_anthropic": langchain_anthropic}):
            result = middleware.modify_model_request(fake_request)
            assert result is fake_request
            assert len(w) == 1
            assert (
                "AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models, not instances of"
                in str(w[-1].message)
            )

    middleware = AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")

    result = middleware.modify_model_request(fake_request)
    assert result is fake_request

    with patch.dict("sys.modules", {"langchain_anthropic": {"ChatAnthropic": object()}}):
        result = middleware.modify_model_request(fake_request)
        assert result is fake_request


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
    result = middleware_disabled.before_model(state)
    assert result is None

    # Test when token count is below threshold
    def mock_token_counter(messages):
        return 500  # Below threshold

    middleware.token_counter = mock_token_counter
    result = middleware.before_model(state)
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
            from langchain_core.messages import AIMessage

            return AIMessage(content="Generated summary")

        def _generate(self, messages, **kwargs):
            from langchain_core.outputs import ChatResult, ChatGeneration
            from langchain_core.messages import AIMessage

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
            from langchain_core.outputs import ChatResult, ChatGeneration
            from langchain_core.messages import AIMessage

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
            from langchain_core.messages import AIMessage

            return AIMessage(content="Generated summary")

        def _generate(self, messages, **kwargs):
            from langchain_core.outputs import ChatResult, ChatGeneration
            from langchain_core.messages import AIMessage

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
    result = middleware.before_model(state)

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


def test_modify_model_request() -> None:
    class ModifyMiddleware(AgentMiddleware):
        def modify_model_request(self, request: ModelRequest, state: AgentState) -> ModelRequest:
            request.messages.append(HumanMessage("remember to be nice!"))
            return request

    builder = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[ModifyMiddleware()],
    )

    agent = builder.compile()
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

    # Compile and invoke the agent
    compiled_agent = agent.compile()
    result = compiled_agent.invoke(
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
    assert "response" in result
    assert result["response"] is not None
    assert hasattr(result["response"], "temperature")
    assert result["response"].temperature == 72.0
    assert result["response"].condition == "sunny"


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
    agent = agent.compile()
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

        def modify_model_request(
            self, request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> ModelRequest:
            assert runtime is not None
            return request

        def after_model(self, state: AgentState, runtime: Runtime) -> None:
            assert runtime is not None
            return None

    middleware = CustomMiddleware()

    agent = create_agent(model=FakeToolCallingModel(), middleware=[CustomMiddleware()])
    agent = agent.compile()
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
    ).compile()

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
