from unittest.mock import patch

import pytest
from langchain_core.language_models import ModelProfile
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from langchain.agents.middleware.summarization import SummarizationMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel


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

    profile: ModelProfile | None = ModelProfile(max_input_tokens=1000)

    @property
    def _llm_type(self) -> str:
        return "mock"


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

    with pytest.raises(
        ValueError,
        match="Model profile information is required to use fractional token limits, "
        "and is unavailable for the specified model",
    ):
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
            msg = "Model error"
            raise ValueError(msg)

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
    middleware.token_counter = len

    trimmed = middleware._trim_messages_for_summary(messages)
    assert trimmed is messages


def test_summarization_middleware_profile_inference_triggers_summary() -> None:
    """Ensure automatic profile inference triggers summarization when limits are exceeded."""

    def token_counter(messages):
        return len(messages) * 200

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
    # we have total_tokens = 4 * 200 = 800
    # and max_input_tokens = 1000
    # since 0.81 * 1000 == 810 > 800 -> summarization not triggered
    result = middleware.before_model(state, None)
    assert result is None

    # Engage summarization
    # since 0.80 * 1000 == 800 <= 800
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


def test_summarization_middleware_token_retention_advances_past_tool_messages() -> None:
    """Ensure token retention advances past tool messages for aggressive summarization."""

    def token_counter(messages: list[AnyMessage]) -> int:
        return sum(len(getattr(message, "content", "")) for message in messages)

    middleware = SummarizationMiddleware(
        model=ProfileChatModel(),
        trigger=("fraction", 0.1),
        keep=("fraction", 0.5),
    )
    middleware.token_counter = token_counter

    # Total tokens: 300 + 200 + 50 + 180 + 160 = 890
    # Target keep: 500 tokens (50% of 1000)
    # Binary search finds cutoff around index 2 (ToolMessage)
    # We advance past it to index 3 (HumanMessage)
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
    # With aggressive summarization, we advance past the ToolMessage
    # So we preserve messages from index 3 onward (the two HumanMessages)
    assert preserved_messages == messages[3:]

    # Verify preserved tokens are within budget
    target_token_count = int(1000 * 0.5)
    preserved_tokens = middleware.token_counter(preserved_messages)
    assert preserved_tokens <= target_token_count


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
            msg = "Profile not available"
            raise ImportError(msg)

    with pytest.raises(
        ValueError,
        match="Model profile information is required to use fractional token limits",
    ):
        _ = SummarizationMiddleware(
            model=ImportErrorProfileModel, trigger=("fraction", 0.5), keep=("messages", 1)
        )


def test_summarization_middleware_full_workflow() -> None:
    """Test SummarizationMiddleware complete summarization workflow."""
    with pytest.warns(DeprecationWarning, match="messages_to_keep is deprecated"):
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
    messages_above = [*messages_at_threshold, HumanMessage(content="6")]
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
        msg = "Token counting failed"
        raise ValueError(msg)

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


def test_summarization_middleware_find_safe_cutoff_point() -> None:
    """Test _find_safe_cutoff_point finds safe cutoff past ToolMessages."""
    model = FakeToolCallingModel()
    middleware = SummarizationMiddleware(
        model=model, trigger=("messages", 10), keep=("messages", 2)
    )

    messages: list[AnyMessage] = [
        HumanMessage(content="msg1"),
        AIMessage(content="ai", tool_calls=[{"name": "tool", "args": {}, "id": "call1"}]),
        ToolMessage(content="result1", tool_call_id="call1"),
        ToolMessage(content="result2", tool_call_id="call2"),
        HumanMessage(content="msg2"),
    ]

    # Starting at a non-ToolMessage returns the same index
    assert middleware._find_safe_cutoff_point(messages, 0) == 0
    assert middleware._find_safe_cutoff_point(messages, 1) == 1

    # Starting at a ToolMessage advances to the next non-ToolMessage
    assert middleware._find_safe_cutoff_point(messages, 2) == 4
    assert middleware._find_safe_cutoff_point(messages, 3) == 4

    # Starting at the HumanMessage after tools returns that index
    assert middleware._find_safe_cutoff_point(messages, 4) == 4

    # Starting past the end returns the index unchanged
    assert middleware._find_safe_cutoff_point(messages, 5) == 5

    # Cutoff at or past length stays the same
    assert middleware._find_safe_cutoff_point(messages, len(messages)) == len(messages)
    assert middleware._find_safe_cutoff_point(messages, len(messages) + 5) == len(messages) + 5


def test_summarization_middleware_zero_and_negative_target_tokens() -> None:
    """Test handling of edge cases with target token calculations."""
    # Test with very small fraction that rounds to zero
    middleware = SummarizationMiddleware(
        model=ProfileChatModel(), trigger=("fraction", 0.0001), keep=("fraction", 0.0001)
    )

    # Should set threshold to 1 when calculated value is <= 0
    messages = [HumanMessage(content="test")]

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
            msg = "Async model error"
            raise ValueError(msg)

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
    with pytest.warns(DeprecationWarning, match="max_tokens_before_summary is deprecated"):
        middleware = SummarizationMiddleware(
            model=MockChatModel(), trigger=("tokens", 2000), max_tokens_before_summary=1000
        )
    assert middleware.trigger == ("tokens", 2000)

    # Test that messages_to_keep is ignored when keep is not default
    with pytest.warns(DeprecationWarning, match="messages_to_keep is deprecated"):
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


def test_summarization_adjust_token_counts() -> None:
    test_message = HumanMessage(content="a" * 12)

    middleware = SummarizationMiddleware(model=MockChatModel(), trigger=("messages", 5))
    count_1 = middleware.token_counter([test_message])

    class MockAnthropicModel(MockChatModel):
        @property
        def _llm_type(self) -> str:
            return "anthropic-chat"

    middleware = SummarizationMiddleware(model=MockAnthropicModel(), trigger=("messages", 5))
    count_2 = middleware.token_counter([test_message])

    assert count_1 != count_2


def test_summarization_middleware_many_parallel_tool_calls_safety() -> None:
    """Test cutoff safety with many parallel tool calls extending beyond old search range."""
    middleware = SummarizationMiddleware(
        model=MockChatModel(), trigger=("messages", 15), keep=("messages", 5)
    )
    tool_calls = [{"name": f"tool_{i}", "args": {}, "id": f"call_{i}"} for i in range(10)]
    human_message = HumanMessage(content="calling 10 tools")
    ai_message = AIMessage(content="calling 10 tools", tool_calls=tool_calls)
    tool_messages = [
        ToolMessage(content=f"result_{i}", tool_call_id=f"call_{i}") for i in range(10)
    ]
    messages: list[AnyMessage] = [human_message, ai_message, *tool_messages]

    # Cutoff at index 7 (a ToolMessage) advances to index 12 (end of messages)
    assert middleware._find_safe_cutoff_point(messages, 7) == 12

    # Any cutoff pointing at a ToolMessage (indices 2-11) advances to index 12
    for i in range(2, 12):
        assert middleware._find_safe_cutoff_point(messages, i) == 12

    # Cutoff at index 0, 1 (before tool messages) stays the same
    assert middleware._find_safe_cutoff_point(messages, 0) == 0
    assert middleware._find_safe_cutoff_point(messages, 1) == 1


def test_summarization_middleware_find_safe_cutoff_advances_past_tools() -> None:
    """Test _find_safe_cutoff advances past ToolMessages to find safe cutoff."""
    middleware = SummarizationMiddleware(
        model=MockChatModel(), trigger=("messages", 10), keep=("messages", 3)
    )

    # Messages list: [Human, AI, Tool, Tool, Tool, Human]
    messages: list[AnyMessage] = [
        HumanMessage(content="msg1"),
        AIMessage(
            content="ai",
            tool_calls=[
                {"name": "tool1", "args": {}, "id": "call1"},
                {"name": "tool2", "args": {}, "id": "call2"},
                {"name": "tool3", "args": {}, "id": "call3"},
            ],
        ),
        ToolMessage(content="result1", tool_call_id="call1"),
        ToolMessage(content="result2", tool_call_id="call2"),
        ToolMessage(content="result3", tool_call_id="call3"),
        HumanMessage(content="msg2"),
    ]

    # Target cutoff index is len(messages) - messages_to_keep = 6 - 3 = 3
    # Index 3 is a ToolMessage, so we advance past the tool sequence to index 5
    cutoff = middleware._find_safe_cutoff(messages, messages_to_keep=3)
    assert cutoff == 5

    # With messages_to_keep=2, target cutoff index is 6 - 2 = 4
    # Index 4 is a ToolMessage, so we advance past the tool sequence to index 5
    # This is aggressive - we keep only 1 message instead of 2
    cutoff = middleware._find_safe_cutoff(messages, messages_to_keep=2)
    assert cutoff == 5


def test_summarization_middleware_cutoff_at_start_of_tool_sequence() -> None:
    """Test cutoff when target lands exactly at the first ToolMessage."""
    middleware = SummarizationMiddleware(
        model=MockChatModel(), trigger=("messages", 8), keep=("messages", 4)
    )

    messages: list[AnyMessage] = [
        HumanMessage(content="msg1"),
        HumanMessage(content="msg2"),
        AIMessage(content="ai", tool_calls=[{"name": "tool", "args": {}, "id": "call1"}]),
        ToolMessage(content="result", tool_call_id="call1"),
        HumanMessage(content="msg3"),
        HumanMessage(content="msg4"),
    ]

    # Target cutoff index is len(messages) - messages_to_keep = 6 - 4 = 2
    # Index 2 is an AIMessage (safe cutoff point), so no adjustment needed
    cutoff = middleware._find_safe_cutoff(messages, messages_to_keep=4)
    assert cutoff == 2


def test_and_trigger_conditions() -> None:
    """Test AND-capable trigger conditions (all conditions in dict must be met)."""
    model = FakeToolCallingModel()

    # Create middleware with AND condition: tokens >= 1000 AND messages >= 5
    middleware = SummarizationMiddleware(
        model=model,
        trigger={"tokens": 1000, "messages": 5},
        keep=("messages", 2),  # Explicitly set a smaller keep value
    )

    # Test case 1: Only tokens threshold met (messages = 3 < 5)
    # Should NOT trigger summarization
    def token_counter_high(messages):
        return 1500  # Above token threshold

    middleware.token_counter = token_counter_high
    state = {
        "messages": [
            HumanMessage(content="1"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
        ]
    }
    result = middleware.before_model(state, None)
    assert result is None, "Should not summarize when only tokens condition is met"

    # Test case 2: Only messages threshold met (tokens = 500 < 1000)
    # Should NOT trigger summarization
    def token_counter_low(messages):
        return 500  # Below token threshold

    middleware.token_counter = token_counter_low
    state = {
        "messages": [
            HumanMessage(content="1"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
            AIMessage(content="4"),
            HumanMessage(content="5"),
            AIMessage(content="6"),
        ]
    }
    result = middleware.before_model(state, None)
    assert result is None, "Should not summarize when only messages condition is met"

    # Test case 3: Both conditions met (tokens >= 1000 AND messages >= 5)
    # Should trigger summarization
    middleware.token_counter = token_counter_high
    result = middleware.before_model(state, None)
    assert result is not None, "Should summarize when both conditions are met"
    assert isinstance(result["messages"][0], RemoveMessage)


def test_or_trigger_conditions_with_and_clauses() -> None:
    """Test OR across multiple AND clauses."""
    model = FakeToolCallingModel()

    # Create middleware with OR of AND conditions:
    # (tokens >= 5000 AND messages >= 3) OR (tokens >= 3000 AND messages >= 6)
    middleware = SummarizationMiddleware(
        model=model,
        trigger=[
            {"tokens": 5000, "messages": 3},
            {"tokens": 3000, "messages": 6},
        ],
    )

    # Test case 1: First clause met (tokens = 5500, messages = 4)
    # Should trigger summarization
    def token_counter_5500(messages):
        return 5500

    middleware.token_counter = token_counter_5500
    state = {
        "messages": [
            HumanMessage(content="1"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
            AIMessage(content="4"),
        ]
    }
    result = middleware.before_model(state, None)
    assert result is not None, "Should summarize when first OR clause is met"

    # Test case 2: Second clause met (tokens = 3500, messages = 7)
    # Should trigger summarization
    def token_counter_3500(messages):
        return 3500

    middleware.token_counter = token_counter_3500
    state = {"messages": [HumanMessage(content=str(i)) for i in range(7)]}
    result = middleware.before_model(state, None)
    assert result is not None, "Should summarize when second OR clause is met"

    # Test case 3: Neither clause fully met
    # (tokens = 4500 meets second token threshold but not message count)
    # (messages = 4 meets first message threshold but not token count)
    # Should NOT trigger summarization
    def token_counter_4500(messages):
        return 4500

    middleware.token_counter = token_counter_4500
    state = {
        "messages": [
            HumanMessage(content="1"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
            AIMessage(content="4"),
        ]
    }
    result = middleware.before_model(state, None)
    assert result is None, "Should not summarize when no complete clause is met"


def test_backward_compatibility_tuple_trigger() -> None:
    """Test backward compatibility with existing tuple-based triggers."""
    model = FakeToolCallingModel()

    # Single tuple trigger
    middleware_single = SummarizationMiddleware(
        model=model,
        trigger=("tokens", 1000),
    )

    def token_counter_high(messages):
        return 1500

    middleware_single.token_counter = token_counter_high
    state = {"messages": [HumanMessage(content="test")]}
    result = middleware_single.before_model(state, None)
    assert result is not None, "Single tuple trigger should work"

    # List of tuples trigger
    middleware_list = SummarizationMiddleware(
        model=model,
        trigger=[("tokens", 1000), ("messages", 5)],
    )

    # Should trigger with high tokens (first condition met)
    middleware_list.token_counter = token_counter_high
    state = {"messages": [HumanMessage(content="test")]}
    result = middleware_list.before_model(state, None)
    assert result is not None, "List of tuples should trigger when any condition met"

    # Should trigger with many messages (second condition met)
    def token_counter_low(messages):
        return 100

    middleware_list.token_counter = token_counter_low
    state = {"messages": [HumanMessage(content=str(i)) for i in range(6)]}
    result = middleware_list.before_model(state, None)
    assert result is not None, "List of tuples should trigger when second condition met"


def test_mixed_and_or_conditions() -> None:
    """Test mixing dict (AND) and tuple (single condition) triggers in a list (OR)."""
    model = FakeToolCallingModel()

    # (tokens >= 4000 AND messages >= 10) OR (messages >= 50)
    middleware = SummarizationMiddleware(
        model=model,
        trigger=[
            {"tokens": 4000, "messages": 10},
            ("messages", 50),
        ],
    )

    # Test case 1: First AND clause met
    def token_counter_high(messages):
        return 4500

    middleware.token_counter = token_counter_high
    state = {"messages": [HumanMessage(content=str(i)) for i in range(12)]}
    result = middleware.before_model(state, None)
    assert result is not None, "Should trigger when AND clause is met"

    # Test case 2: Second simple condition met
    def token_counter_low(messages):
        return 1000

    middleware.token_counter = token_counter_low
    state = {"messages": [HumanMessage(content=str(i)) for i in range(55)]}
    result = middleware.before_model(state, None)
    assert result is not None, "Should trigger when simple messages condition is met"

    # Test case 3: Neither condition met
    middleware.token_counter = token_counter_low
    state = {"messages": [HumanMessage(content=str(i)) for i in range(8)]}
    result = middleware.before_model(state, None)
    assert result is None, "Should not trigger when no condition is met"


def test_fraction_in_and_trigger() -> None:
    """Test using fraction threshold in AND conditions."""
    # Create middleware with AND condition: fraction >= 0.8 AND messages >= 5
    middleware = SummarizationMiddleware(
        model=ProfileChatModel(),
        trigger={"fraction": 0.8, "messages": 5},
    )

    def token_counter(messages):
        return len(messages) * 200  # Each message = 200 tokens

    middleware.token_counter = token_counter

    # Test case 1: Both conditions met
    # 5 messages * 200 = 1000 tokens (profile max is 1000)
    # 1000 / 1000 = 1.0 >= 0.8  AND messages = 5 >= 5
    state = {"messages": [HumanMessage(content=str(i)) for i in range(5)]}
    result = middleware.before_model(state, None)
    assert result is not None, "Should trigger when both fraction and messages conditions met"

    # Test case 2: Only messages condition met
    # 3 messages * 200 = 600 tokens
    # 600 / 1000 = 0.6 < 0.8 and messages = 3 < 5
    state = {"messages": [HumanMessage(content=str(i)) for i in range(3)]}
    result = middleware.before_model(state, None)
    assert result is None, "Should not trigger when neither condition is fully met"

    # Test case 3: High fraction but not enough messages
    # 4 messages * 200 = 800 tokens
    # 800 / 1000 = 0.8 >= 0.8 but messages = 4 < 5
    state = {"messages": [HumanMessage(content=str(i)) for i in range(4)]}
    result = middleware.before_model(state, None)
    assert result is None, "Should not trigger when only fraction condition is met"


def test_trigger_validation_errors() -> None:
    """Test validation errors for invalid trigger configurations."""
    model = FakeToolCallingModel()

    # Invalid metric name
    with pytest.raises(ValueError, match="Unsupported trigger metric"):
        SummarizationMiddleware(
            model=model,
            trigger={"invalid_metric": 100},
        )

    # Invalid fraction value (> 1)
    with pytest.raises(ValueError, match="fraction must be > 0 and <= 1"):
        SummarizationMiddleware(
            model=model,
            trigger={"fraction": 1.5},
        )

    # Invalid fraction value (<= 0)
    with pytest.raises(ValueError, match="fraction must be > 0 and <= 1"):
        SummarizationMiddleware(
            model=model,
            trigger={"fraction": 0},
        )

    # Invalid token threshold (<= 0)
    with pytest.raises(ValueError, match="tokens threshold must be > 0"):
        SummarizationMiddleware(
            model=model,
            trigger={"tokens": 0},
        )

    # Invalid message threshold (<= 0)
    with pytest.raises(ValueError, match="messages threshold must be > 0"):
        SummarizationMiddleware(
            model=model,
            trigger={"messages": -5},
        )

    # Non-numeric fraction value
    with pytest.raises(ValueError, match="Fraction trigger values must be numeric"):
        SummarizationMiddleware(
            model=model,
            trigger={"fraction": "invalid"},
        )

    # Invalid list item type
    with pytest.raises(TypeError, match="Unsupported trigger item type"):
        SummarizationMiddleware(
            model=model,
            trigger=["invalid"],
        )


def test_empty_and_condition() -> None:
    """Test that empty dict trigger clause is rejected or handled appropriately."""
    model = FakeToolCallingModel()

    # Empty dict should be allowed but never triggers (no conditions to check)
    middleware = SummarizationMiddleware(
        model=model,
        trigger={},
    )

    def token_counter_high(messages):
        return 5000

    middleware.token_counter = token_counter_high
    state = {"messages": [HumanMessage(content=str(i)) for i in range(100)]}
    # Empty clause should vacuously be true (all zero conditions are met)
    result = middleware.before_model(state, None)
    assert result is not None, "Empty trigger clause should trigger"
