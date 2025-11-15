from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from langchain.agents.middleware.summarization import SummarizationMiddleware

from ...model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain_model_profiles import ModelProfile


class MockChatModel(BaseChatModel):
    """Mock chat model for testing."""

    def invoke(self, prompt):  # type: ignore[no-untyped-def]
        return AIMessage(content="Generated summary")

    def _generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

    @property
    def _llm_type(self) -> str:
        return "mock"


class ProfileChatModel(BaseChatModel):
    """Mock chat model with profile for testing."""

    def _generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

    @property
    def _llm_type(self) -> str:
        return "mock"

    @property
    def profile(self) -> "ModelProfile":
        return {"max_input_tokens": 1000}


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
    middleware = SummarizationMiddleware(model=MockChatModel(), trigger=("tokens", 1000))

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
        SummarizationMiddleware(model=MockChatModel(), max_tokens_before_summary=500)
    with pytest.warns(DeprecationWarning, match="messages_to_keep is deprecated"):
        SummarizationMiddleware(model=MockChatModel(), messages_to_keep=5)


def test_summarization_middleware_trim_limit_none_keeps_all_messages() -> None:
    """Verify disabling trim limit preserves full message sequence."""
    messages = [HumanMessage(content=str(i)) for i in range(10)]
    middleware = SummarizationMiddleware(
        model=MockChatModel(),
        trim_tokens_to_summarize=None,
    )
    middleware.token_counter = lambda msgs: len(msgs)

    trimmed = middleware._trim_messages_for_summary(messages)
    assert trimmed is messages


def test_summarization_middleware_profile_inference_triggers_summary() -> None:
    """Ensure automatic profile inference triggers summarization when limits are exceeded."""
    token_counter = lambda messages: len(messages) * 200

    middleware = SummarizationMiddleware(
        model=ProfileChatModel(),
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
        model=ProfileChatModel(),
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
        model=ProfileChatModel(),
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
        model=ProfileChatModel(),
        trigger=("fraction", 0.80),
        keep=("fraction", 0.8),
        token_counter=token_counter,
    )
    assert middleware.before_model(state, None) is None

    # Test with tokens_to_keep as absolute int value
    middleware_int = SummarizationMiddleware(
        model=ProfileChatModel(),
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
        model=ProfileChatModel(),
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

    def token_counter(messages: list[AnyMessage]) -> int:
        return sum(len(getattr(message, "content", "")) for message in messages)

    middleware = SummarizationMiddleware(
        model=ProfileChatModel(),
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
    with pytest.warns(DeprecationWarning):
        # keep test for functionality
        middleware = SummarizationMiddleware(
            model=MockChatModel(), max_tokens_before_summary=1000, messages_to_keep=2
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
    # Test that summarization is triggered when message count reaches threshold
    middleware = SummarizationMiddleware(
        model=MockChatModel(), trigger=("messages", 5), keep=("messages", 2)
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
    middleware_disabled = SummarizationMiddleware(model=MockChatModel(), trigger=None)
    result = middleware_disabled.before_model(state_above, None)
    assert result is None


@pytest.mark.parametrize(
    ("param_name", "param_value", "expected_error"),
    [
        ("trigger", ("fraction", 0.0), "Fractional trigger values must be between 0 and 1"),
        ("trigger", ("fraction", 1.5), "Fractional trigger values must be between 0 and 1"),
        ("keep", ("fraction", -0.1), "Fractional keep values must be between 0 and 1"),
        ("trigger", ("tokens", 0), "trigger thresholds must be greater than 0"),
        ("trigger", ("messages", -5), "trigger thresholds must be greater than 0"),
        ("keep", ("tokens", 0), "keep thresholds must be greater than 0"),
        ("trigger", ("invalid", 100), "Unsupported context size type"),
        ("keep", ("invalid", 100), "Unsupported context size type"),
    ],
)
def test_summarization_middleware_validation_edge_cases(
    param_name: str, param_value: tuple[str, float | int], expected_error: str
) -> None:
    """Test validation of context size parameters with edge cases."""
    model = FakeToolCallingModel()
    with pytest.raises(ValueError, match=expected_error):
        SummarizationMiddleware(model=model, **{param_name: param_value})


def test_summarization_middleware_multiple_triggers() -> None:
    """Test middleware with multiple trigger conditions."""
    # Test with multiple triggers - should activate when ANY condition is met
    middleware = SummarizationMiddleware(
        model=MockChatModel(),
        trigger=[("messages", 10), ("tokens", 500)],
        keep=("messages", 2),
    )

    # Mock token counter to return low count
    def mock_low_tokens(messages):
        return 100

    middleware.token_counter = mock_low_tokens

    # Should not trigger - neither condition met
    messages = [HumanMessage(content=str(i)) for i in range(5)]
    state = {"messages": messages}
    result = middleware.before_model(state, None)
    assert result is None

    # Should trigger - message count threshold met
    messages = [HumanMessage(content=str(i)) for i in range(10)]
    state = {"messages": messages}
    result = middleware.before_model(state, None)
    assert result is not None

    # Test token trigger
    def mock_high_tokens(messages):
        return 600

    middleware.token_counter = mock_high_tokens
    messages = [HumanMessage(content=str(i)) for i in range(5)]
    state = {"messages": messages}
    result = middleware.before_model(state, None)
    assert result is not None


def test_summarization_middleware_profile_edge_cases() -> None:
    """Test profile retrieval with various edge cases."""

    class NoProfileModel(BaseChatModel):
        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        @property
        def _llm_type(self):
            return "mock"

    # Model without profile attribute
    middleware = SummarizationMiddleware(model=NoProfileModel(), trigger=("messages", 5))
    assert middleware._get_profile_limits() is None

    class InvalidProfileModel(BaseChatModel):
        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        @property
        def _llm_type(self):
            return "mock"

        @property
        def profile(self):
            return "invalid_profile_type"

    # Model with non-dict profile
    middleware = SummarizationMiddleware(model=InvalidProfileModel(), trigger=("messages", 5))
    assert middleware._get_profile_limits() is None

    class MissingTokensModel(BaseChatModel):
        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        @property
        def _llm_type(self):
            return "mock"

        @property
        def profile(self):
            return {"other_field": 100}

    # Model with profile but no max_input_tokens
    middleware = SummarizationMiddleware(model=MissingTokensModel(), trigger=("messages", 5))
    assert middleware._get_profile_limits() is None

    class InvalidTokenTypeModel(BaseChatModel):
        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        @property
        def _llm_type(self):
            return "mock"

        @property
        def profile(self):
            return {"max_input_tokens": "not_an_int"}

    # Model with non-int max_input_tokens
    middleware = SummarizationMiddleware(model=InvalidTokenTypeModel(), trigger=("messages", 5))
    assert middleware._get_profile_limits() is None


def test_summarization_middleware_trim_messages_error_fallback() -> None:
    """Test that trim_messages_for_summary falls back gracefully on errors."""
    middleware = SummarizationMiddleware(model=MockChatModel(), trigger=("messages", 5))

    # Create a mock token counter that raises an exception
    def failing_token_counter(messages):
        raise Exception("Token counting failed")

    middleware.token_counter = failing_token_counter

    # Should fall back to last 15 messages
    messages = [HumanMessage(content=str(i)) for i in range(20)]
    trimmed = middleware._trim_messages_for_summary(messages)
    assert len(trimmed) == 15
    assert trimmed == messages[-15:]


def test_summarization_middleware_binary_search_edge_cases() -> None:
    """Test binary search in _find_token_based_cutoff with edge cases."""
    middleware = SummarizationMiddleware(
        model=MockChatModel(), trigger=("messages", 5), keep=("tokens", 100)
    )

    # Test with single message that's too large
    def token_counter_single_large(messages):
        return len(messages) * 200

    middleware.token_counter = token_counter_single_large

    single_message = [HumanMessage(content="x" * 200)]
    cutoff = middleware._find_token_based_cutoff(single_message)
    assert cutoff == 0

    # Test with empty messages
    cutoff = middleware._find_token_based_cutoff([])
    assert cutoff == 0

    # Test when all messages fit within token budget
    def token_counter_small(messages):
        return len(messages) * 10

    middleware.token_counter = token_counter_small
    messages = [HumanMessage(content=str(i)) for i in range(5)]
    cutoff = middleware._find_token_based_cutoff(messages)
    assert cutoff == 0


def test_summarization_middleware_tool_call_extraction_edge_cases() -> None:
    """Test tool call ID extraction with various message formats."""
    model = FakeToolCallingModel()
    middleware = SummarizationMiddleware(model=model, trigger=("messages", 5))

    # Test with dict-style tool calls
    ai_message_dict = AIMessage(
        content="test", tool_calls=[{"name": "tool1", "args": {}, "id": "id1"}]
    )
    ids = middleware._extract_tool_call_ids(ai_message_dict)
    assert ids == {"id1"}

    # Test with multiple tool calls
    ai_message_multiple = AIMessage(
        content="test",
        tool_calls=[
            {"name": "tool1", "args": {}, "id": "id1"},
            {"name": "tool2", "args": {}, "id": "id2"},
        ],
    )
    ids = middleware._extract_tool_call_ids(ai_message_multiple)
    assert ids == {"id1", "id2"}

    # Test with empty tool calls list
    ai_message_empty = AIMessage(content="test", tool_calls=[])
    ids = middleware._extract_tool_call_ids(ai_message_empty)
    assert len(ids) == 0


def test_summarization_middleware_complex_tool_pair_scenarios() -> None:
    """Test complex tool call pairing scenarios."""
    model = FakeToolCallingModel()
    middleware = SummarizationMiddleware(model=model, trigger=("messages", 5), keep=("messages", 3))

    # Test with multiple AI messages with tool calls
    messages = [
        HumanMessage(content="msg1"),
        AIMessage(content="ai1", tool_calls=[{"name": "tool1", "args": {}, "id": "call1"}]),
        ToolMessage(content="result1", tool_call_id="call1"),
        HumanMessage(content="msg2"),
        AIMessage(content="ai2", tool_calls=[{"name": "tool2", "args": {}, "id": "call2"}]),
        ToolMessage(content="result2", tool_call_id="call2"),
        HumanMessage(content="msg3"),
    ]

    # Test cutoff at index 1 - unsafe (separates first AI/Tool pair)
    assert not middleware._is_safe_cutoff_point(messages, 2)

    # Test cutoff at index 3 - safe (keeps first pair together)
    assert middleware._is_safe_cutoff_point(messages, 3)

    # Test cutoff at index 5 - unsafe (separates second AI/Tool pair)
    assert not middleware._is_safe_cutoff_point(messages, 5)

    # Test _cutoff_separates_tool_pair directly
    assert middleware._cutoff_separates_tool_pair(messages, 1, 2, {"call1"})
    assert not middleware._cutoff_separates_tool_pair(messages, 1, 0, {"call1"})
    assert not middleware._cutoff_separates_tool_pair(messages, 1, 3, {"call1"})


def test_summarization_middleware_tool_call_in_search_range() -> None:
    """Test tool call safety with messages at edge of search range."""
    model = FakeToolCallingModel()
    middleware = SummarizationMiddleware(
        model=model, trigger=("messages", 10), keep=("messages", 2)
    )

    # Create messages with tool pair separated by some distance
    # Search range is 5, so messages within 5 positions of cutoff are checked
    messages = [
        HumanMessage(content="msg1"),
        HumanMessage(content="msg2"),
        AIMessage(content="ai", tool_calls=[{"name": "tool", "args": {}, "id": "call1"}]),
        HumanMessage(content="msg3"),
        HumanMessage(content="msg4"),
        ToolMessage(content="result", tool_call_id="call1"),
        HumanMessage(content="msg6"),
    ]

    # Cutoff at index 3 would separate: [0,1,2] from [3,4,5,6]
    # AI at index 2 is before cutoff, Tool at index 5 is after cutoff - unsafe
    assert not middleware._is_safe_cutoff_point(messages, 3)

    # Cutoff at index 6 keeps AI and Tool both in summarized section
    assert middleware._is_safe_cutoff_point(messages, 6)

    # Cutoff at index 0 or 1 also safe - both AI and Tool in preserved section
    assert middleware._is_safe_cutoff_point(messages, 0)
    assert middleware._is_safe_cutoff_point(messages, 1)


def test_summarization_middleware_zero_and_negative_target_tokens() -> None:
    """Test handling of edge cases with target token calculations."""
    # Test with very small fraction that rounds to zero
    middleware = SummarizationMiddleware(
        model=ProfileChatModel(), trigger=("fraction", 0.0001), keep=("fraction", 0.0001)
    )

    # Should set threshold to 1 when calculated value is <= 0
    messages = [HumanMessage(content="test")]
    state = {"messages": messages}

    # The trigger fraction calculation: int(1000 * 0.0001) = 0, but should be set to 1
    # Token count of 1 message should exceed threshold of 1
    def token_counter(msgs):
        return 2

    middleware.token_counter = token_counter
    assert middleware._should_summarize(messages, 2)


async def test_summarization_middleware_async_error_handling() -> None:
    """Test async summary creation with errors."""

    class ErrorAsyncModel(BaseChatModel):
        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Summary"))])

        async def _agenerate(self, messages, **kwargs):
            raise Exception("Async model error")

        @property
        def _llm_type(self):
            return "mock"

    middleware = SummarizationMiddleware(model=ErrorAsyncModel(), trigger=("messages", 5))
    messages = [HumanMessage(content="test")]
    summary = await middleware._acreate_summary(messages)
    assert "Error generating summary: Async model error" in summary


def test_summarization_middleware_cutoff_at_boundary() -> None:
    """Test cutoff index determination at exact message boundaries."""
    middleware = SummarizationMiddleware(
        model=MockChatModel(), trigger=("messages", 5), keep=("messages", 5)
    )

    # When we want to keep exactly as many messages as we have
    messages = [HumanMessage(content=str(i)) for i in range(5)]
    cutoff = middleware._find_safe_cutoff(messages, 5)
    assert cutoff == 0  # Should not cut anything

    # When we want to keep more messages than we have
    cutoff = middleware._find_safe_cutoff(messages, 10)
    assert cutoff == 0


def test_summarization_middleware_deprecated_parameters_with_defaults() -> None:
    """Test that deprecated parameters work correctly with default values."""
    # Test that deprecated max_tokens_before_summary is ignored when trigger is set
    with pytest.warns(DeprecationWarning):
        middleware = SummarizationMiddleware(
            model=MockChatModel(), trigger=("tokens", 2000), max_tokens_before_summary=1000
        )
    assert middleware.trigger == ("tokens", 2000)

    # Test that messages_to_keep is ignored when keep is not default
    with pytest.warns(DeprecationWarning):
        middleware = SummarizationMiddleware(
            model=MockChatModel(), keep=("messages", 5), messages_to_keep=10
        )
    assert middleware.keep == ("messages", 5)


def test_summarization_middleware_fraction_trigger_with_no_profile() -> None:
    """Test fractional trigger condition when profile data becomes unavailable."""
    middleware = SummarizationMiddleware(
        model=ProfileChatModel(),
        trigger=[("fraction", 0.5), ("messages", 100)],
        keep=("messages", 5),
    )

    # Test that when fractional condition can't be evaluated, other triggers still work
    messages = [HumanMessage(content=str(i)) for i in range(100)]

    # Mock _get_profile_limits to return None
    original_method = middleware._get_profile_limits
    middleware._get_profile_limits = lambda: None

    # Should still trigger based on message count
    state = {"messages": messages}
    result = middleware.before_model(state, None)
    assert result is not None

    # Restore original method
    middleware._get_profile_limits = original_method


def test_summarization_middleware_is_safe_cutoff_at_end() -> None:
    """Test _is_safe_cutoff_point when cutoff is at or past the end."""
    model = FakeToolCallingModel()
    middleware = SummarizationMiddleware(model=model, trigger=("messages", 5))

    messages = [HumanMessage(content=str(i)) for i in range(5)]

    # Cutoff at exactly the length should be safe
    assert middleware._is_safe_cutoff_point(messages, len(messages))

    # Cutoff past the length should also be safe
    assert middleware._is_safe_cutoff_point(messages, len(messages) + 5)
