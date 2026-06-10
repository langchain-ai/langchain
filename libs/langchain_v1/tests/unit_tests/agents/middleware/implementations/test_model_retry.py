"""Tests for ModelRetryMiddleware functionality."""

import time
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field
from typing_extensions import override

from langchain.agents.factory import create_agent
from langchain.agents.middleware._retry import calculate_delay
from langchain.agents.middleware.model_retry import ModelRetryMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from tests.unit_tests.agents.model import FakeToolCallingModel


class TemporaryFailureModel(FakeToolCallingModel):
    """Model that fails a certain number of times before succeeding."""

    fail_count: int = Field(default=0)
    attempt: int = Field(default=0)

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Execute the model.

        Args:
            messages: Input messages.
            stop: Optional stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatResult with success message if attempt >= fail_count.

        Raises:
            ValueError: If attempt < fail_count.
        """
        self.attempt += 1
        if self.attempt <= self.fail_count:
            msg = f"Temporary failure {self.attempt}"
            raise ValueError(msg)
        # Return success message
        ai_msg = AIMessage(content=f"Success after {self.attempt} attempts", id=str(self.index))
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


class AlwaysFailingModel(FakeToolCallingModel):
    """Model that always fails with a specific exception."""

    error_message: str = Field(default="Model error")
    error_type: type[Exception] = Field(default=ValueError)

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Execute the model and raise exception.

        Args:
            messages: Input messages.
            stop: Optional stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Additional keyword arguments.

        Raises:
            Exception: Always raises the configured exception.
        """
        raise self.error_type(self.error_message)


def test_model_retry_initialization_defaults() -> None:
    """Test ModelRetryMiddleware initialization with default values."""
    retry = ModelRetryMiddleware()

    assert retry.max_retries == 2
    assert retry.tools == []
    assert retry.on_failure == "continue"
    assert retry.backoff_factor == 2.0
    assert retry.initial_delay == 1.0
    assert retry.max_delay == 60.0
    assert retry.jitter is True


def test_model_retry_initialization_custom() -> None:
    """Test ModelRetryMiddleware initialization with custom values."""
    retry = ModelRetryMiddleware(
        max_retries=5,
        retry_on=(ValueError, RuntimeError),
        on_failure="error",
        backoff_factor=1.5,
        initial_delay=0.5,
        max_delay=30.0,
        jitter=False,
    )

    assert retry.max_retries == 5
    assert retry.tools == []
    assert retry.retry_on == (ValueError, RuntimeError)
    assert retry.on_failure == "error"
    assert retry.backoff_factor == 1.5
    assert retry.initial_delay == 0.5
    assert retry.max_delay == 30.0
    assert retry.jitter is False


def test_model_retry_invalid_max_retries() -> None:
    """Test ModelRetryMiddleware raises error for invalid max_retries."""
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        ModelRetryMiddleware(max_retries=-1)


def test_model_retry_invalid_initial_delay() -> None:
    """Test ModelRetryMiddleware raises error for invalid initial_delay."""
    with pytest.raises(ValueError, match="initial_delay must be >= 0"):
        ModelRetryMiddleware(initial_delay=-1.0)


def test_model_retry_invalid_max_delay() -> None:
    """Test ModelRetryMiddleware raises error for invalid max_delay."""
    with pytest.raises(ValueError, match="max_delay must be >= 0"):
        ModelRetryMiddleware(max_delay=-1.0)


def test_model_retry_invalid_backoff_factor() -> None:
    """Test ModelRetryMiddleware raises error for invalid backoff_factor."""
    with pytest.raises(ValueError, match="backoff_factor must be >= 0"):
        ModelRetryMiddleware(backoff_factor=-1.0)


def test_model_retry_working_model_no_retry_needed() -> None:
    """Test ModelRetryMiddleware with a working model (no retry needed)."""
    model = FakeToolCallingModel()

    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0.01, jitter=False)

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    assert "Hello" in ai_messages[-1].content


def test_model_retry_failing_model_returns_message() -> None:
    """Test ModelRetryMiddleware with failing model returns error message."""
    model = AlwaysFailingModel(error_message="Model error", error_type=ValueError)

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    # Should contain error message with attempts
    last_msg = ai_messages[-1].content
    assert "failed after 3 attempts" in last_msg
    assert "ValueError" in last_msg


def test_model_retry_failing_model_raises() -> None:
    """Test ModelRetryMiddleware with on_failure='error' re-raises exception."""
    model = AlwaysFailingModel(error_message="Model error", error_type=ValueError)

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        on_failure="error",
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    # Should raise the ValueError from the model
    with pytest.raises(ValueError, match="Model error"):
        agent.invoke(
            {"messages": [HumanMessage("Hello")]},
            {"configurable": {"thread_id": "test"}},
        )


def test_model_retry_custom_failure_formatter() -> None:
    """Test ModelRetryMiddleware with custom failure message formatter."""

    def custom_formatter(exc: Exception) -> str:
        return f"Custom error: {type(exc).__name__}"

    model = AlwaysFailingModel(error_message="Model error", error_type=ValueError)

    retry = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0.01,
        jitter=False,
        on_failure=custom_formatter,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    assert "Custom error: ValueError" in ai_messages[-1].content


def test_model_retry_succeeds_after_retries() -> None:
    """Test ModelRetryMiddleware succeeds after temporary failures."""
    model = TemporaryFailureModel(fail_count=2)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.01,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    # Should succeed on 3rd attempt
    assert "Success after 3 attempts" in ai_messages[-1].content
    assert model.attempt == 3


def test_model_retry_specific_exceptions() -> None:
    """Test ModelRetryMiddleware only retries specific exception types."""
    # This model will fail with RuntimeError, which we won't retry
    model = AlwaysFailingModel(error_message="Runtime error", error_type=RuntimeError)

    # Only retry ValueError
    retry = ModelRetryMiddleware(
        max_retries=2,
        retry_on=(ValueError,),
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    # RuntimeError should fail immediately (1 attempt only)
    assert "1 attempt" in ai_messages[-1].content


def test_model_retry_custom_exception_filter() -> None:
    """Test ModelRetryMiddleware with custom exception filter function."""

    class CustomError(Exception):
        """Custom exception with retry_me attribute."""

        def __init__(self, message: str, *, retry_me: bool):
            """Initialize custom error.

            Args:
                message: Error message.
                retry_me: Whether this error should be retried.
            """
            super().__init__(message)
            self.retry_me = retry_me

    attempt_count = {"value": 0}

    class CustomErrorModel(FakeToolCallingModel):
        """Model that raises CustomError."""

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Execute the model and raise CustomError.

            Args:
                messages: Input messages.
                stop: Optional stop sequences.
                run_manager: Optional callback manager.
                **kwargs: Additional keyword arguments.

            Raises:
                CustomError: Always raises CustomError.
            """
            attempt_count["value"] += 1
            if attempt_count["value"] == 1:
                msg = "Retryable error"
                raise CustomError(msg, retry_me=True)
            msg = "Non-retryable error"
            raise CustomError(msg, retry_me=False)

    def should_retry(exc: Exception) -> bool:
        return isinstance(exc, CustomError) and exc.retry_me

    model = CustomErrorModel()

    retry = ModelRetryMiddleware(
        max_retries=3,
        retry_on=should_retry,
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1

    # Should retry once (attempt 1 with retry_me=True), then fail on attempt 2 (retry_me=False)
    assert attempt_count["value"] == 2
    assert "2 attempts" in ai_messages[-1].content


def test_model_retry_backoff_timing() -> None:
    """Test ModelRetryMiddleware applies correct backoff delays."""
    model = TemporaryFailureModel(fail_count=3)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    start_time = time.time()
    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )
    elapsed = time.time() - start_time

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1

    # Expected delays: 0.1 + 0.2 + 0.4 = 0.7 seconds
    # Allow some margin for execution time
    assert elapsed >= 0.6, f"Expected at least 0.6s, got {elapsed}s"


def test_model_retry_constant_backoff() -> None:
    """Test ModelRetryMiddleware with constant backoff (backoff_factor=0)."""
    model = TemporaryFailureModel(fail_count=2)

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.1,
        backoff_factor=0.0,  # Constant backoff
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    start_time = time.time()
    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )
    elapsed = time.time() - start_time

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1

    # Expected delays: 0.1 + 0.1 = 0.2 seconds (constant)
    assert elapsed >= 0.15, f"Expected at least 0.15s, got {elapsed}s"
    assert elapsed < 0.5, f"Expected less than 0.5s (exponential would be longer), got {elapsed}s"


def test_model_retry_max_delay_cap() -> None:
    """Test calculate_delay caps delay at max_delay."""
    # Test delay calculation with aggressive backoff and max_delay cap
    delay_0 = calculate_delay(
        0,
        backoff_factor=10.0,  # Very aggressive backoff
        initial_delay=1.0,
        max_delay=2.0,  # Cap at 2 seconds
        jitter=False,
    )  # 1.0
    delay_1 = calculate_delay(
        1,
        backoff_factor=10.0,
        initial_delay=1.0,
        max_delay=2.0,
        jitter=False,
    )  # 10.0 -> capped to 2.0
    delay_2 = calculate_delay(
        2,
        backoff_factor=10.0,
        initial_delay=1.0,
        max_delay=2.0,
        jitter=False,
    )  # 100.0 -> capped to 2.0

    assert delay_0 == 1.0
    assert delay_1 == 2.0
    assert delay_2 == 2.0


def test_model_retry_jitter_variation() -> None:
    """Test calculate_delay adds jitter to delays."""
    # Generate multiple delays and ensure they vary
    delays = [
        calculate_delay(
            0,
            backoff_factor=1.0,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=True,
        )
        for _ in range(10)
    ]

    # All delays should be within ±25% of 1.0 (i.e., between 0.75 and 1.25)
    for delay in delays:
        assert 0.75 <= delay <= 1.25

    # Delays should vary (not all the same)
    assert len(set(delays)) > 1


@pytest.mark.asyncio
async def test_model_retry_async_working_model() -> None:
    """Test ModelRetryMiddleware with async execution and working model."""
    model = FakeToolCallingModel()

    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0.01, jitter=False)

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    assert "Hello" in ai_messages[-1].content


@pytest.mark.asyncio
async def test_model_retry_async_failing_model() -> None:
    """Test ModelRetryMiddleware with async execution and failing model."""
    model = AlwaysFailingModel(error_message="Model error", error_type=ValueError)

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    last_msg = ai_messages[-1].content
    assert "failed after 3 attempts" in last_msg
    assert "ValueError" in last_msg


@pytest.mark.asyncio
async def test_model_retry_async_succeeds_after_retries() -> None:
    """Test ModelRetryMiddleware async execution succeeds after temporary failures."""
    model = TemporaryFailureModel(fail_count=2)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.01,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    assert "Success after 3 attempts" in ai_messages[-1].content


@pytest.mark.asyncio
async def test_model_retry_async_backoff_timing() -> None:
    """Test ModelRetryMiddleware async applies correct backoff delays."""
    model = TemporaryFailureModel(fail_count=3)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    start_time = time.time()
    result = await agent.ainvoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )
    elapsed = time.time() - start_time

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1

    # Expected delays: 0.1 + 0.2 + 0.4 = 0.7 seconds
    assert elapsed >= 0.6, f"Expected at least 0.6s, got {elapsed}s"


def test_model_retry_zero_retries() -> None:
    """Test ModelRetryMiddleware with max_retries=0 (no retries)."""
    model = AlwaysFailingModel(error_message="Model error", error_type=ValueError)

    retry = ModelRetryMiddleware(
        max_retries=0,  # No retries
        on_failure="continue",
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    # Should fail after 1 attempt (no retries)
    assert "1 attempt" in ai_messages[-1].content


def test_model_retry_multiple_middleware_composition() -> None:
    """Test ModelRetryMiddleware composes correctly with other middleware."""
    call_log = []

    # Custom middleware that logs calls
    @wrap_model_call
    def logging_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        call_log.append("before_model")
        response = handler(request)
        call_log.append("after_model")
        return response

    model = FakeToolCallingModel()

    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0.01, jitter=False)

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[logging_middleware, retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Both middleware should be called
    assert call_log == ["before_model", "after_model"]

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    assert "Hello" in ai_messages[-1].content
