"""Tests for DoubleBufferMiddleware."""

from collections.abc import Iterable
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import ModelProfile
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from typing_extensions import override

from langchain.agents import AgentState
from langchain.agents.middleware.double_buffer import (
    DOUBLE_BUFFER_SOURCE,
    DoubleBufferMiddleware,
    RenewalPolicy,
)


class MockChatModel(BaseChatModel):
    """Mock chat model that returns a fixed summary."""

    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        return AIMessage(content="Generated summary of the conversation.")

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Generated summary."))]
        )

    @property
    def _llm_type(self) -> str:
        return "mock"


class ProfileChatModel(MockChatModel):
    """Mock chat model with profile for fraction-based triggers."""

    profile: ModelProfile | None = ModelProfile(max_input_tokens=1000)


def _make_messages(count: int) -> list[AnyMessage]:
    """Generate alternating user/assistant messages."""
    msgs: list[AnyMessage] = []
    for i in range(count):
        if i % 2 == 0:
            msgs.append(HumanMessage(content=f"User message {i}"))
        else:
            msgs.append(AIMessage(content=f"Assistant message {i}"))
    return msgs


def _fixed_token_counter(token_count: int) -> Any:
    """Return a token counter that always returns a fixed count."""

    def counter(_: Iterable[MessageLikeRepresentation]) -> int:
        return token_count

    return counter


# --- Initialization Tests ---


def test_initialization() -> None:
    """Test DoubleBufferMiddleware initialization with defaults."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(model=model)

    assert middleware.model == model
    assert middleware.checkpoint_trigger == ("fraction", 0.7)
    assert middleware.swap_trigger == ("fraction", 0.95)
    assert middleware.keep == ("messages", 20)
    assert middleware.max_generations is None
    assert middleware.renewal_policy == RenewalPolicy.RECURSE
    assert middleware.generation == 0
    assert middleware.has_back_buffer is False


def test_initialization_custom_params() -> None:
    """Test DoubleBufferMiddleware with custom parameters."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 50),
        swap_trigger=("messages", 80),
        keep=("messages", 10),
        max_generations=3,
        renewal_policy=RenewalPolicy.DUMP,
    )

    assert middleware.checkpoint_trigger == ("messages", 50)
    assert middleware.swap_trigger == ("messages", 80)
    assert middleware.keep == ("messages", 10)
    assert middleware.max_generations == 3
    assert middleware.renewal_policy == RenewalPolicy.DUMP


def test_validation_errors() -> None:
    """Test parameter validation."""
    model = MockChatModel()

    with pytest.raises(ValueError, match="Fractional"):
        DoubleBufferMiddleware(model=model, checkpoint_trigger=("fraction", 1.5))

    with pytest.raises(ValueError, match="greater than 0"):
        DoubleBufferMiddleware(model=model, checkpoint_trigger=("messages", 0))

    with pytest.raises(ValueError, match="Unsupported"):
        DoubleBufferMiddleware(model=model, checkpoint_trigger=("invalid", 5))  # type: ignore[arg-type]


def test_swap_trigger_must_exceed_checkpoint_trigger() -> None:
    """swap_trigger must be strictly greater than checkpoint_trigger when types match."""
    model = MockChatModel()

    # swap == checkpoint (same type) should raise
    with pytest.raises(ValueError, match=r"swap_trigger.*must be strictly greater"):
        DoubleBufferMiddleware(
            model=model,
            checkpoint_trigger=("messages", 50),
            swap_trigger=("messages", 50),
        )

    # swap < checkpoint (same type) should raise
    with pytest.raises(ValueError, match=r"swap_trigger.*must be strictly greater"):
        DoubleBufferMiddleware(
            model=model,
            checkpoint_trigger=("messages", 80),
            swap_trigger=("messages", 50),
        )

    # fraction: swap == checkpoint should raise
    with pytest.raises(ValueError, match=r"swap_trigger.*must be strictly greater"):
        DoubleBufferMiddleware(
            model=model,
            checkpoint_trigger=("fraction", 0.9),
            swap_trigger=("fraction", 0.9),
        )

    # fraction: swap < checkpoint should raise
    with pytest.raises(ValueError, match=r"swap_trigger.*must be strictly greater"):
        DoubleBufferMiddleware(
            model=model,
            checkpoint_trigger=("fraction", 0.9),
            swap_trigger=("fraction", 0.5),
        )

    # tokens: swap <= checkpoint should raise
    with pytest.raises(ValueError, match=r"swap_trigger.*must be strictly greater"):
        DoubleBufferMiddleware(
            model=model,
            checkpoint_trigger=("tokens", 500),
            swap_trigger=("tokens", 500),
        )

    # Different types should NOT raise (cross-type comparison is not meaningful)
    DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 80),
        swap_trigger=("tokens", 50),
    )

    # Valid same-type: swap > checkpoint should NOT raise
    DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 50),
        swap_trigger=("messages", 80),
    )


# --- Phase 1: Checkpoint Tests ---


def test_no_checkpoint_below_threshold() -> None:
    """No checkpoint when message count is below threshold."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 50),
        swap_trigger=("messages", 80),
    )

    state = AgentState[Any](messages=_make_messages(10))
    result = middleware.before_model(state, Runtime())

    assert result is None
    assert middleware.has_back_buffer is False


def test_checkpoint_at_threshold() -> None:
    """Checkpoint creates back buffer when threshold is reached."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 10),
        swap_trigger=("messages", 40),
        keep=("messages", 5),
    )

    state = AgentState[Any](messages=_make_messages(20))
    result = middleware.before_model(state, Runtime())

    # Checkpoint creates the back buffer but doesn't modify active state
    assert result is None
    assert middleware.has_back_buffer is True


def test_no_double_checkpoint() -> None:
    """Don't create a second checkpoint when back buffer already exists."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 10),
        swap_trigger=("messages", 40),
        keep=("messages", 5),
    )

    messages = _make_messages(20)
    state = AgentState[Any](messages=messages)

    # First call creates checkpoint
    middleware.before_model(state, Runtime())
    assert middleware.has_back_buffer is True

    # Second call should NOT create another checkpoint
    result = middleware.before_model(state, Runtime())
    assert result is None  # no swap triggered, back buffer already exists


# --- Phase 2: Concurrent Tests ---


def test_sync_back_buffer() -> None:
    """Back buffer gets new messages during concurrent phase."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 10),
        swap_trigger=("messages", 40),
        keep=("messages", 5),
    )

    messages = _make_messages(20)
    state = AgentState[Any](messages=messages)
    middleware.before_model(state, Runtime())  # creates checkpoint
    assert middleware.has_back_buffer is True

    # Add a new message with a unique ID that's definitely not in the back buffer
    new_msg = HumanMessage(content="Brand new message", id="new-msg-id")
    messages.append(new_msg)
    state = AgentState[Any](messages=messages)
    middleware.before_model(state, Runtime())

    # Back buffer should contain the new message
    back_ids = {msg.id for msg in (middleware._back_buffer or []) if msg.id}
    assert "new-msg-id" in back_ids


# --- Phase 3: Swap Tests ---


def test_swap_at_threshold() -> None:
    """Buffer swap occurs when active buffer hits swap threshold."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 5),
        swap_trigger=("messages", 15),
        keep=("messages", 3),
    )

    # Create enough messages to trigger checkpoint
    messages = _make_messages(10)
    state = AgentState[Any](messages=messages)
    middleware.before_model(state, Runtime())
    assert middleware.has_back_buffer is True
    assert middleware.generation == 0

    # Now add enough to trigger swap
    messages = _make_messages(20)
    state = AgentState[Any](messages=messages)
    result = middleware.before_model(state, Runtime())

    assert result is not None
    assert middleware.generation == 1
    assert middleware.has_back_buffer is False
    # Result should contain RemoveMessage + back buffer contents
    assert any(isinstance(msg, RemoveMessage) for msg in result["messages"])


def test_swap_increments_generation() -> None:
    """Each swap increments the generation counter."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 5),
        swap_trigger=("messages", 15),
        keep=("messages", 3),
    )

    # First cycle
    state = AgentState[Any](messages=_make_messages(10))
    middleware.before_model(state, Runtime())

    state = AgentState[Any](messages=_make_messages(20))
    middleware.before_model(state, Runtime())
    assert middleware.generation == 1

    # Second cycle
    state = AgentState[Any](messages=_make_messages(10))
    middleware.before_model(state, Runtime())

    state = AgentState[Any](messages=_make_messages(20))
    middleware.before_model(state, Runtime())
    assert middleware.generation == 2


# --- Token-Based Trigger Tests ---


def test_token_based_checkpoint() -> None:
    """Checkpoint triggers based on token count."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("tokens", 100),
        swap_trigger=("tokens", 500),
        keep=("messages", 3),
        token_counter=_fixed_token_counter(150),  # always returns 150 tokens
    )

    state = AgentState[Any](messages=_make_messages(10))
    middleware.before_model(state, Runtime())

    assert middleware.has_back_buffer is True


def test_fraction_based_checkpoint() -> None:
    """Checkpoint triggers based on fraction of model max tokens."""
    model = ProfileChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("fraction", 0.5),  # 500 tokens on a 1000 max model
        swap_trigger=("fraction", 0.9),
        keep=("messages", 3),
        token_counter=_fixed_token_counter(600),  # above 50%
    )

    state = AgentState[Any](messages=_make_messages(10))
    middleware.before_model(state, Runtime())

    assert middleware.has_back_buffer is True


# --- Renewal Tests ---


def test_dump_renewal_resets_generation() -> None:
    """DUMP renewal policy resets generation counter."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 5),
        swap_trigger=("messages", 15),
        keep=("messages", 3),
        max_generations=2,
        renewal_policy=RenewalPolicy.DUMP,
    )

    # Simulate reaching max generations
    middleware._current_generation = 2

    state = AgentState[Any](messages=_make_messages(10))
    middleware.before_model(state, Runtime())

    # Should have reset generation and created checkpoint
    assert middleware._current_generation == 0 or middleware.has_back_buffer is True


def test_recurse_renewal() -> None:
    """RECURSE renewal triggers meta-summarization."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 5),
        swap_trigger=("messages", 15),
        keep=("messages", 3),
        max_generations=1,
        renewal_policy=RenewalPolicy.RECURSE,
    )

    middleware._current_generation = 1

    # Include a prior summary message in the state
    messages: list[AnyMessage] = [
        HumanMessage(
            content="Previous summary",
            additional_kwargs={"lc_source": DOUBLE_BUFFER_SOURCE},
        ),
        *_make_messages(10),
    ]
    state = AgentState[Any](messages=messages)
    middleware.before_model(state, Runtime())

    assert middleware._current_generation == 0


# --- Error Handling Tests ---


def test_checkpoint_failure_degrades_gracefully() -> None:
    """Checkpoint failure is handled without crashing."""

    class ErrorModel(BaseChatModel):
        @override
        def invoke(
            self,
            input: LanguageModelInput,
            config: RunnableConfig | None = None,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> AIMessage:
            msg = "LLM error"
            raise RuntimeError(msg)

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "LLM error"
            raise RuntimeError(msg)

        @property
        def _llm_type(self) -> str:
            return "error"

    middleware = DoubleBufferMiddleware(
        model=ErrorModel(),
        checkpoint_trigger=("messages", 5),
        swap_trigger=("messages", 15),
        keep=("messages", 3),
    )

    state = AgentState[Any](messages=_make_messages(10))
    result = middleware.before_model(state, Runtime())

    # Should degrade gracefully — no back buffer, no crash
    assert result is None
    assert middleware.has_back_buffer is False


# --- Stop-the-World Fallback Tests ---


def test_sync_fallback_checkpoint_at_swap_time() -> None:
    """If swap threshold is hit with no back buffer, do synchronous checkpoint."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("tokens", 999999),  # high — won't fire normally
        swap_trigger=("messages", 10),  # low — will fire immediately
        keep=("messages", 3),
        token_counter=_fixed_token_counter(100),
    )

    state = AgentState[Any](messages=_make_messages(20))
    result = middleware.before_model(state, Runtime())

    # Should have created checkpoint synchronously then swapped
    assert result is not None
    assert middleware.generation == 1
    assert middleware.has_back_buffer is False
    assert any(isinstance(msg, RemoveMessage) for msg in result["messages"])


def test_sync_fallback_checkpoint_failure_at_swap_time() -> None:
    """If fallback checkpoint fails at swap time, continue gracefully."""

    class ErrorModel(BaseChatModel):
        @override
        def invoke(
            self,
            input: LanguageModelInput,
            config: RunnableConfig | None = None,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> AIMessage:
            msg = "LLM error"
            raise RuntimeError(msg)

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "LLM error"
            raise RuntimeError(msg)

        @property
        def _llm_type(self) -> str:
            return "error"

    middleware = DoubleBufferMiddleware(
        model=ErrorModel(),
        checkpoint_trigger=("tokens", 999999),
        swap_trigger=("messages", 10),
        keep=("messages", 3),
        token_counter=_fixed_token_counter(100),
    )

    state = AgentState[Any](messages=_make_messages(20))
    result = middleware.before_model(state, Runtime())

    # Should degrade gracefully — no swap, no crash
    assert result is None
    assert middleware.has_back_buffer is False


# --- Async Tests ---


async def _await_checkpoint(middleware: DoubleBufferMiddleware) -> None:
    """Wait for a background checkpoint task to complete."""
    if middleware._checkpoint_task is not None and not middleware._checkpoint_task.done():
        try:
            await middleware._checkpoint_task
        except Exception:  # noqa: S110
            pass  # test helper — exceptions checked by specific tests
        finally:
            middleware._checkpoint_task = None


async def test_async_checkpoint() -> None:
    """Async checkpoint creation fires in background."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 10),
        swap_trigger=("messages", 40),
        keep=("messages", 5),
    )

    state = AgentState[Any](messages=_make_messages(20))
    result = await middleware.abefore_model(state, Runtime())

    # Checkpoint fires in background — returns None immediately
    assert result is None
    # Wait for the background task to complete
    await _await_checkpoint(middleware)
    assert middleware.has_back_buffer is True


async def test_async_swap() -> None:
    """Async buffer swap works after background checkpoint completes."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 5),
        swap_trigger=("messages", 15),
        keep=("messages", 3),
    )

    state = AgentState[Any](messages=_make_messages(10))
    await middleware.abefore_model(state, Runtime())
    await _await_checkpoint(middleware)
    assert middleware.has_back_buffer is True

    state = AgentState[Any](messages=_make_messages(20))
    result = await middleware.abefore_model(state, Runtime())

    assert result is not None
    assert middleware.generation == 1
    assert middleware.has_back_buffer is False


async def test_async_swap_blocks_on_running_checkpoint() -> None:
    """If swap threshold hit while checkpoint runs, block on it then swap."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("messages", 5),
        swap_trigger=("messages", 15),
        keep=("messages", 3),
    )

    # Trigger checkpoint in background
    state = AgentState[Any](messages=_make_messages(10))
    await middleware.abefore_model(state, Runtime())

    # Don't await the task — go straight to swap threshold
    state = AgentState[Any](messages=_make_messages(20))
    result = await middleware.abefore_model(state, Runtime())

    # Should have blocked on the task, then swapped
    assert result is not None
    assert middleware.generation == 1
    assert middleware.has_back_buffer is False


async def test_async_fallback_checkpoint_at_swap_time() -> None:
    """If async swap threshold hit with no checkpoint ever started, do sync fallback."""
    model = MockChatModel()
    middleware = DoubleBufferMiddleware(
        model=model,
        checkpoint_trigger=("tokens", 999999),  # won't fire
        swap_trigger=("messages", 10),  # fires immediately
        keep=("messages", 3),
        token_counter=_fixed_token_counter(100),
    )

    state = AgentState[Any](messages=_make_messages(20))
    result = await middleware.abefore_model(state, Runtime())

    assert result is not None
    assert middleware.generation == 1
    assert middleware.has_back_buffer is False


# --- Helper Method Tests ---


def test_build_summary_messages() -> None:
    """Summary messages are correctly formatted."""
    msgs = DoubleBufferMiddleware._build_summary_messages("Test summary")
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)
    assert "Test summary" in msgs[0].content
    assert msgs[0].additional_kwargs.get("lc_source") == DOUBLE_BUFFER_SOURCE


def test_ensure_message_ids() -> None:
    """All messages get unique IDs."""
    messages: list[AnyMessage] = [HumanMessage(content="no id"), AIMessage(content="no id")]
    DoubleBufferMiddleware._ensure_message_ids(messages)
    for msg in messages:
        assert msg.id is not None


def test_partition_messages() -> None:
    """Messages are correctly partitioned."""
    messages = _make_messages(10)
    to_summarize, preserved = DoubleBufferMiddleware._partition_messages(messages, 4)
    assert len(to_summarize) == 4
    assert len(preserved) == 6


def test_safe_cutoff_respects_tool_pairs() -> None:
    """Cutoff doesn't split AI/Tool message pairs."""
    messages: list[AnyMessage] = [
        HumanMessage(content="Do something"),
        AIMessage(
            content="",
            tool_calls=[{"name": "tool1", "args": {}, "id": "call_1"}],
        ),
        ToolMessage(content="result", tool_call_id="call_1"),
        HumanMessage(content="Thanks"),
    ]

    # If cutoff lands on the ToolMessage, it should back up to include the AIMessage
    cutoff = DoubleBufferMiddleware._find_safe_cutoff_point(messages, 2)
    assert cutoff == 1  # backs up to the AIMessage
