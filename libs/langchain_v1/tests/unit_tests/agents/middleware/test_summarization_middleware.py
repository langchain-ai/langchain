from typing import TYPE_CHECKING

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from unittest.mock import patch

from langchain.agents.middleware.summarization import SummarizationMiddleware

from ..model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain_model_profiles import ModelProfile


def test_summarization_middleware_initialization() -> None:
    """Test SummarizationMiddleware initialization."""
    model = FakeToolCallingModel()
    middleware = SummarizationMiddleware(
        model=model,
        trigger=("tokens", 1000),
        keep=("messages", 10),
        summary_prompt="Custom prompt: {messages}",
    )

    assert middleware.model == model
    assert middleware.trigger == ("tokens", 1000)
    assert middleware.keep == ("messages", 10)
    assert middleware.summary_prompt == "Custom prompt: {messages}"
    assert middleware.trim_tokens_to_summarize == 4000

    with pytest.raises(ValueError):
        SummarizationMiddleware(model=model, keep=("fraction", 0.5))  # no model profile

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
    middleware = SummarizationMiddleware(model=model, trigger=("tokens", 1000))

    # Test when summarization is disabled
    middleware_disabled = SummarizationMiddleware(model=model, trigger=None)
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
    middleware = SummarizationMiddleware(model=model, trigger=("tokens", 1000))

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
        model=model, trigger=("tokens", 1000), keep=("messages", 3)
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

    middleware = SummarizationMiddleware(model=MockModel(), trigger=("tokens", 1000))

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

    middleware_error = SummarizationMiddleware(model=ErrorModel(), trigger=("tokens", 1000))
    summary = middleware_error._create_summary(messages)
    assert "Error generating summary: Model error" in summary

    # Test we raise warning if max_tokens_before_summary or messages_to_keep is specified
    with pytest.warns(DeprecationWarning, match="max_tokens_before_summary is deprecated"):
        SummarizationMiddleware(model=MockModel(), max_tokens_before_summary=500)
    with pytest.warns(DeprecationWarning, match="messages_to_keep is deprecated"):
        SummarizationMiddleware(model=MockModel(), messages_to_keep=5)


def test_summarization_middleware_trim_limit_none_keeps_all_messages() -> None:
    """Verify disabling trim limit preserves full message sequence."""

    class MockModel(BaseChatModel):
        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        @property
        def _llm_type(self):
            return "mock"

    messages = [HumanMessage(content=str(i)) for i in range(10)]
    middleware = SummarizationMiddleware(
        model=MockModel(),
        trim_tokens_to_summarize=None,
    )
    middleware.token_counter = lambda msgs: len(msgs)

    trimmed = middleware._trim_messages_for_summary(messages)
    assert trimmed is messages


def test_summarization_middleware_profile_inference_triggers_summary() -> None:
    """Ensure automatic profile inference triggers summarization when limits are exceeded."""

    class ProfileModel(BaseChatModel):
        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        @property
        def _llm_type(self) -> str:
            return "mock"

        @property
        def profile(self) -> "ModelProfile":
            return {"max_input_tokens": 1000}

    token_counter = lambda messages: len(messages) * 200

    middleware = SummarizationMiddleware(
        model=ProfileModel(),
        trigger=("fraction", 0.81),
        keep=("fraction", 0.5),
        token_counter=token_counter,
    )

    state = {
        "messages": [
            HumanMessage(content="Message 1"),
            AIMessage(content="Message 2"),
            HumanMessage(content="Message 3"),
            AIMessage(content="Message 4"),
        ]
    }

    # Test we don't engage summarization
    # total_tokens = 4 * 200 = 800
    # max_input_tokens = 1000
    # 0.81 * 1000 == 810 > 800 -> summarization not triggered
    result = middleware.before_model(state, None)
    assert result is None

    # Engage summarization
    # 0.80 * 1000 == 800 <= 800
    middleware = SummarizationMiddleware(
        model=ProfileModel(),
        trigger=("fraction", 0.80),
        keep=("fraction", 0.5),
        token_counter=token_counter,
    )
    result = middleware.before_model(state, None)
    assert result is not None
    assert isinstance(result["messages"][0], RemoveMessage)
    summary_message = result["messages"][1]
    assert isinstance(summary_message, HumanMessage)
    assert summary_message.text.startswith("Here is a summary of the conversation")
    assert len(result["messages"][2:]) == 2  # Preserved messages
    assert [message.content for message in result["messages"][2:]] == [
        "Message 3",
        "Message 4",
    ]

    # With keep=("fraction", 0.6) the target token allowance becomes 600,
    # so the cutoff shifts to keep the last three messages instead of two.
    middleware = SummarizationMiddleware(
        model=ProfileModel(),
        trigger=("fraction", 0.80),
        keep=("fraction", 0.6),
        token_counter=token_counter,
    )
    result = middleware.before_model(state, None)
    assert result is not None
    assert [message.content for message in result["messages"][2:]] == [
        "Message 2",
        "Message 3",
        "Message 4",
    ]

    # Once keep=("fraction", 0.8) the inferred limit equals the full
    # context (target tokens = 800), so token-based retention keeps everything
    # and summarization is skipped entirely.
    middleware = SummarizationMiddleware(
        model=ProfileModel(),
        trigger=("fraction", 0.80),
        keep=("fraction", 0.8),
        token_counter=token_counter,
    )
    assert middleware.before_model(state, None) is None

    # Test with tokens_to_keep as absolute int value
    middleware_int = SummarizationMiddleware(
        model=ProfileModel(),
        trigger=("fraction", 0.80),
        keep=("tokens", 400),  # Keep exactly 400 tokens (2 messages)
        token_counter=token_counter,
    )
    result = middleware_int.before_model(state, None)
    assert result is not None
    assert [message.content for message in result["messages"][2:]] == [
        "Message 3",
        "Message 4",
    ]

    # Test with tokens_to_keep as larger int value
    middleware_int_large = SummarizationMiddleware(
        model=ProfileModel(),
        trigger=("fraction", 0.80),
        keep=("tokens", 600),  # Keep 600 tokens (3 messages)
        token_counter=token_counter,
    )
    result = middleware_int_large.before_model(state, None)
    assert result is not None
    assert [message.content for message in result["messages"][2:]] == [
        "Message 2",
        "Message 3",
        "Message 4",
    ]


def test_summarization_middleware_token_retention_pct_respects_tool_pairs() -> None:
    """Ensure token retention keeps pairs together even if exceeding target tokens."""

    class ProfileModel(BaseChatModel):
        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        @property
        def _llm_type(self) -> str:
            return "mock"

        @property
        def profile(self) -> "ModelProfile":
            return {"max_input_tokens": 1000}

    def token_counter(messages):
        return sum(len(getattr(message, "content", "")) for message in messages)

    middleware = SummarizationMiddleware(
        model=ProfileModel(),
        trigger=("fraction", 0.1),
        keep=("fraction", 0.5),
    )
    middleware.token_counter = token_counter

    messages: list[AnyMessage] = [
        HumanMessage(content="H" * 300),
        AIMessage(
            content="A" * 200,
            tool_calls=[{"name": "test", "args": {}, "id": "call-1"}],
        ),
        ToolMessage(content="T" * 50, tool_call_id="call-1"),
        HumanMessage(content="H" * 180),
        HumanMessage(content="H" * 160),
    ]

    state = {"messages": messages}
    result = middleware.before_model(state, None)
    assert result is not None

    preserved_messages = result["messages"][2:]
    assert preserved_messages == messages[1:]

    target_token_count = int(1000 * 0.5)
    preserved_tokens = middleware.token_counter(preserved_messages)

    # Tool pair retention can exceed the target token count but should keep the pair intact.
    assert preserved_tokens > target_token_count


def test_summarization_middleware_missing_profile() -> None:
    """Ensure automatic profile inference falls back when profiles are unavailable."""

    class ImportErrorProfileModel(BaseChatModel):
        def _generate(self, messages, **kwargs):
            raise NotImplementedError

        @property
        def _llm_type(self) -> str:
            return "mock"

        @property
        def profile(self):
            raise ImportError("Profile not available")

    with pytest.raises(
        ValueError,
        match="Model profile information is required to use fractional token limits",
    ):
        _ = SummarizationMiddleware(
            model=ImportErrorProfileModel, trigger=("fraction", 0.5), keep=("messages", 1)
        )


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

    with pytest.warns(DeprecationWarning):
        # keep test for functionality
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


async def test_summarization_middleware_full_workflow_async() -> None:
    """Test SummarizationMiddleware complete summarization workflow."""

    class MockModel(BaseChatModel):
        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Blep"))])

        async def _agenerate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Blip"))])

        @property
        def _llm_type(self):
            return "mock"

    middleware = SummarizationMiddleware(
        model=MockModel(), trigger=("tokens", 1000), keep=("messages", 2)
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
    result = await middleware.abefore_model(state, None)

    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) > 0

    expected_types = ["remove", "human", "human", "human"]
    actual_types = [message.type for message in result["messages"]]
    assert actual_types == expected_types
    assert [message.content for message in result["messages"][2:]] == ["4", "5"]

    summary_message = result["messages"][1]
    assert "Blip" in summary_message.text


def test_summarization_middleware_keep_messages() -> None:
    """Test SummarizationMiddleware with keep parameter specifying messages."""

    class MockModel(BaseChatModel):
        def invoke(self, prompt):
            return AIMessage(content="Generated summary")

        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        @property
        def _llm_type(self):
            return "mock"

    # Test that summarization is triggered when message count reaches threshold
    middleware = SummarizationMiddleware(
        model=MockModel(), trigger=("messages", 5), keep=("messages", 2)
    )

    # Below threshold - no summarization
    messages_below = [
        HumanMessage(content="1"),
        HumanMessage(content="2"),
        HumanMessage(content="3"),
        HumanMessage(content="4"),
    ]
    state_below = {"messages": messages_below}
    result = middleware.before_model(state_below, None)
    assert result is None

    # At threshold - should trigger summarization
    messages_at_threshold = [
        HumanMessage(content="1"),
        HumanMessage(content="2"),
        HumanMessage(content="3"),
        HumanMessage(content="4"),
        HumanMessage(content="5"),
    ]
    state_at = {"messages": messages_at_threshold}
    result = middleware.before_model(state_at, None)
    assert result is not None
    assert "messages" in result
    expected_types = ["remove", "human", "human", "human"]
    actual_types = [message.type for message in result["messages"]]
    assert actual_types == expected_types
    assert [message.content for message in result["messages"][2:]] == ["4", "5"]

    # Above threshold - should also trigger summarization
    messages_above = messages_at_threshold + [HumanMessage(content="6")]
    state_above = {"messages": messages_above}
    result = middleware.before_model(state_above, None)
    assert result is not None
    assert "messages" in result
    expected_types = ["remove", "human", "human", "human"]
    actual_types = [message.type for message in result["messages"]]
    assert actual_types == expected_types
    assert [message.content for message in result["messages"][2:]] == ["5", "6"]

    # Test with both parameters disabled
    middleware_disabled = SummarizationMiddleware(model=MockModel(), trigger=None)
    result = middleware_disabled.before_model(state_above, None)
    assert result is None
