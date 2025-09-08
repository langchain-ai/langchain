import pytest
from typing import Any
from unittest.mock import patch

from syrupy.assertion import SnapshotAssertion

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import tool

from langchain.agents.middleware_agent import create_agent
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain.agents.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt.interrupt import ActionRequest, HumanInterruptConfig

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
    tool_configs = {
        "test_tool": HumanInterruptConfig(
            allow_ignore=True, allow_respond=True, allow_edit=True, allow_accept=True
        )
    }

    middleware = HumanInTheLoopMiddleware(tool_configs=tool_configs, message_prefix="Custom prefix")

    assert middleware.tool_configs == tool_configs
    assert middleware.message_prefix == "Custom prefix"


def test_human_in_the_loop_middleware_no_interrupts_needed() -> None:
    """Test HumanInTheLoopMiddleware when no interrupts are needed."""
    tool_configs = {
        "test_tool": HumanInterruptConfig(
            allow_ignore=True, allow_respond=True, allow_edit=True, allow_accept=True
        )
    }

    middleware = HumanInTheLoopMiddleware(tool_configs=tool_configs)

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


def test_human_in_the_loop_middleware_interrupt_responses() -> None:
    """Test HumanInTheLoopMiddleware with different interrupt response types."""
    tool_configs = {
        "test_tool": HumanInterruptConfig(
            allow_ignore=True, allow_respond=True, allow_edit=True, allow_accept=True
        )
    }

    middleware = HumanInTheLoopMiddleware(tool_configs=tool_configs)

    ai_message = AIMessage(
        content="I'll help you",
        tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "1"}],
    )
    state = {"messages": [HumanMessage(content="Hello"), ai_message]}

    # Test accept response
    def mock_accept(requests):
        return [{"type": "accept", "args": None}]

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_accept):
        result = middleware.after_model(state)
        assert result is not None
        assert result["messages"][0] == ai_message
        assert result["messages"][0].tool_calls == ai_message.tool_calls

    # Test edit response
    def mock_edit(requests):
        return [
            {"type": "edit", "args": ActionRequest(action="test_tool", args={"input": "edited"})}
        ]

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_edit):
        result = middleware.after_model(state)
        assert result is not None
        assert result["messages"][0].tool_calls[0]["args"] == {"input": "edited"}

    # Test ignore response
    def mock_ignore(requests):
        return [{"type": "ignore", "args": None}]

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_ignore):
        result = middleware.after_model(state)
        assert result is not None
        assert result["jump_to"] == "__end__"

    # Test response type
    def mock_response(requests):
        return [{"type": "response", "args": "Custom response"}]

    with patch(
        "langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_response
    ):
        result = middleware.after_model(state)
        assert result is not None
        assert result["jump_to"] == "model"
        assert result["messages"][0]["role"] == "tool"
        assert result["messages"][0]["content"] == "Custom response"

    # Test unknown response type
    def mock_unknown(requests):
        return [{"type": "unknown", "args": None}]

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_unknown):
        with pytest.raises(ValueError, match="Unknown response type: unknown"):
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
