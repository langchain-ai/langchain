"""Tests for ModelRetryMiddleware functionality."""

import asyncio
import time
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.factory import create_agent
from langchain.agents.middleware.model_retry import ModelRetryMiddleware


class TemporaryFailureModel(GenericFakeChatModel):
    """Model that fails a certain number of times before succeeding."""

    def __init__(self, fail_count: int, *args, **kwargs):
        """Initialize with the number of times to fail.

        Args:
            fail_count: Number of times to fail before succeeding.
        """
        super().__init__(*args, **kwargs)
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "fail_count", fail_count)
        object.__setattr__(self, "attempt", 0)

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
            stop: Optional list of stop words.
            run_manager: Optional callback manager.
            **kwargs: Additional arguments.

        Returns:
            Success message if attempt >= fail_count.

        Raises:
            ValueError: If attempt < fail_count.
        """
        # Use object.__getattribute__ to access the attribute
        attempt = object.__getattribute__(self, "attempt") + 1
        object.__setattr__(self, "attempt", attempt)
        fail_count = object.__getattribute__(self, "fail_count")

        if attempt <= fail_count:
            msg = f"Temporary failure {attempt}"
            raise ValueError(msg)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class AlwaysFailModel(GenericFakeChatModel):
    """Model that always fails."""

    def _generate(self, messages, **kwargs):
        """Always raise an exception."""
        raise ValueError("Always fails")


def test_model_retry_initialization_defaults() -> None:
    """Test ModelRetryMiddleware initialization with default values."""
    retry = ModelRetryMiddleware()

    assert retry.max_retries == 2
    assert retry.retry_on == (Exception,)
    assert retry.backoff_factor == 2.0
    assert retry.initial_delay == 1.0
    assert retry.max_delay == 60.0
    assert retry.jitter is True


def test_model_retry_initialization_custom() -> None:
    """Test ModelRetryMiddleware initialization with custom values."""
    retry = ModelRetryMiddleware(
        max_retries=5,
        retry_on=(ValueError, RuntimeError),
        backoff_factor=1.5,
        initial_delay=0.5,
        max_delay=30.0,
        jitter=False,
    )

    assert retry.max_retries == 5
    assert retry.retry_on == (ValueError, RuntimeError)
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
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Success")]))

    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0.01, jitter=False)

    agent = create_agent(
        model=model,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Test")]},
        {"configurable": {"thread_id": "test"}},
    )

    assert len(result["messages"]) == 2
    assert result["messages"][1].content == "Success"


def test_model_retry_temporary_failure_succeeds() -> None:
    """Test ModelRetryMiddleware with model that fails then succeeds."""
    # Use external state dictionary (LangChain pattern)
    attempt_count = {"value": 0}
    fail_count = 2

    class TemporaryFailureModel(GenericFakeChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            attempt_count["value"] += 1
            if attempt_count["value"] <= fail_count:
                msg = f"Temporary failure {attempt_count['value']}"
                raise ValueError(msg)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    model = TemporaryFailureModel(
        messages=iter([AIMessage(content="Success after retries")]),
    )

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.01,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Test")]},
        {"configurable": {"thread_id": "test"}},
    )

    assert attempt_count["value"] == 3  # Initial + 2 retries
    assert result["messages"][1].content == "Success after retries"


def test_model_retry_exhausted_raises_exception() -> None:
    """Test ModelRetryMiddleware raises exception when retries exhausted."""
    model = AlwaysFailModel(messages=iter([]))

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(ValueError, match="Always fails"):
        agent.invoke(
            {"messages": [HumanMessage("Test")]},
            {"configurable": {"thread_id": "test"}},
        )


def test_model_retry_specific_exceptions() -> None:
    """Test ModelRetryMiddleware only retries specific exception types."""

    class ValueErrorModel(GenericFakeChatModel):
        def _generate(self, messages, **kwargs):
            raise ValueError("ValueError occurred")

    class RuntimeErrorModel(GenericFakeChatModel):
        def _generate(self, messages, **kwargs):
            raise RuntimeError("RuntimeError occurred")

    # Only retry ValueError
    retry = ModelRetryMiddleware(
        max_retries=2,
        retry_on=(ValueError,),
        initial_delay=0.01,
        jitter=False,
    )

    # ValueError should be retried
    value_error_model = ValueErrorModel(messages=iter([]))
    agent1 = create_agent(
        model=value_error_model,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(ValueError, match="ValueError occurred"):
        agent1.invoke(
            {"messages": [HumanMessage("Test")]},
            {"configurable": {"thread_id": "test1"}},
        )

    # RuntimeError should fail immediately (not retried)
    runtime_error_model = RuntimeErrorModel(messages=iter([]))
    agent2 = create_agent(
        model=runtime_error_model,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(RuntimeError, match="RuntimeError occurred"):
        agent2.invoke(
            {"messages": [HumanMessage("Test")]},
            {"configurable": {"thread_id": "test2"}},
        )


def test_model_retry_custom_exception_filter() -> None:
    """Test ModelRetryMiddleware with custom exception filter function."""

    class CustomError(Exception):
        def __init__(self, status_code: int):
            self.status_code = status_code
            super().__init__(f"Error {status_code}")

    # Use external state (LangChain pattern)
    status_code_storage = {"value": 500}

    class CustomErrorModel(GenericFakeChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            raise CustomError(status_code_storage["value"])

    def should_retry(exc: Exception) -> bool:
        # Only retry on 5xx errors
        if isinstance(exc, CustomError):
            return 500 <= exc.status_code < 600
        return False

    retry = ModelRetryMiddleware(
        max_retries=2,
        retry_on=should_retry,
        initial_delay=0.01,
        jitter=False,
    )

    # 500 error should be retried
    status_code_storage["value"] = 500
    model_500 = CustomErrorModel(messages=iter([]))
    agent1 = create_agent(
        model=model_500,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(CustomError):
        agent1.invoke(
            {"messages": [HumanMessage("Test")]},
            {"configurable": {"thread_id": "test1"}},
        )

    # 400 error should fail immediately (not retried)
    status_code_storage["value"] = 400
    model_400 = CustomErrorModel(messages=iter([]))
    agent2 = create_agent(
        model=model_400,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(CustomError):
        agent2.invoke(
            {"messages": [HumanMessage("Test")]},
            {"configurable": {"thread_id": "test2"}},
        )


def test_model_retry_backoff_delay() -> None:
    """Test ModelRetryMiddleware applies backoff delay correctly."""
    call_times = []

    class DelayTrackingModel(GenericFakeChatModel):
        def _generate(self, messages, **kwargs):
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Temporary failure")
            return super()._generate(messages, **kwargs)

    model = DelayTrackingModel(messages=iter([AIMessage(content="Success")]))

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    start_time = time.time()
    result = agent.invoke(
        {"messages": [HumanMessage("Test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Check that delays were applied
    assert len(call_times) == 3
    # First retry should be ~0.1s after initial call
    assert call_times[1] - call_times[0] >= 0.09  # Allow small timing variance
    # Second retry should be ~0.2s after first retry (2x backoff)
    assert call_times[2] - call_times[1] >= 0.19  # Allow small timing variance
    assert result["messages"][1].content == "Success"


def test_model_retry_constant_backoff() -> None:
    """Test ModelRetryMiddleware with constant backoff (backoff_factor=0)."""
    call_times = []

    class DelayTrackingModel(GenericFakeChatModel):
        def _generate(self, messages, **kwargs):
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Temporary failure")
            return super()._generate(messages, **kwargs)

    model = DelayTrackingModel(messages=iter([AIMessage(content="Success")]))

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.1,
        backoff_factor=0.0,  # Constant delay
        jitter=False,
    )

    agent = create_agent(
        model=model,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Check that constant delays were applied
    assert len(call_times) == 3
    # Both delays should be approximately the same (~0.1s)
    delay1 = call_times[1] - call_times[0]
    delay2 = call_times[2] - call_times[1]
    assert abs(delay1 - delay2) < 0.05  # Delays should be similar
    assert delay1 >= 0.09  # Should be close to 0.1s
    assert result["messages"][1].content == "Success"


def test_model_retry_max_delay_cap() -> None:
    """Test ModelRetryMiddleware respects max_delay cap."""
    call_times = []

    class DelayTrackingModel(GenericFakeChatModel):
        def _generate(self, messages, **kwargs):
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Temporary failure")
            return super()._generate(messages, **kwargs)

    model = DelayTrackingModel(messages=iter([AIMessage(content="Success")]))

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=10.0,  # Would be 1s, 10s, 100s without cap
        max_delay=2.0,  # Cap at 2s
        jitter=False,
    )

    agent = create_agent(
        model=model,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Check that delays were capped
    assert len(call_times) == 3
    delay1 = call_times[1] - call_times[0]
    delay2 = call_times[2] - call_times[1]
    # Both delays should be capped at max_delay (2s)
    assert delay1 <= 2.1  # Allow small timing variance
    assert delay2 <= 2.1  # Allow small timing variance
    assert result["messages"][1].content == "Success"


@pytest.mark.asyncio
async def test_model_retry_async() -> None:
    """Test ModelRetryMiddleware with async agent invocation."""
    # Use external state dictionary (LangChain pattern)
    attempt_count = {"value": 0}
    fail_count = 2

    class TemporaryFailureModel(GenericFakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            attempt_count["value"] += 1
            if attempt_count["value"] <= fail_count:
                msg = f"Temporary failure {attempt_count['value']}"
                raise ValueError(msg)
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    model = TemporaryFailureModel(
        messages=iter([AIMessage(content="Async success")]),
    )

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.01,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Test")]},
        {"configurable": {"thread_id": "test"}},
    )

    assert attempt_count["value"] == 3  # Initial + 2 retries
    assert result["messages"][1].content == "Async success"


@pytest.mark.asyncio
async def test_model_retry_async_backoff() -> None:
    """Test ModelRetryMiddleware applies backoff delay in async context."""
    call_times = []

    class DelayTrackingModel(GenericFakeChatModel):
        async def _agenerate(self, messages, **kwargs):
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Temporary failure")
            return await super()._agenerate(messages, **kwargs)

    model = DelayTrackingModel(messages=iter([AIMessage(content="Success")]))

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    start_time = time.time()
    result = await agent.ainvoke(
        {"messages": [HumanMessage("Test")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Check that delays were applied
    assert len(call_times) == 3
    assert call_times[1] - call_times[0] >= 0.09
    assert call_times[2] - call_times[1] >= 0.19
    assert result["messages"][1].content == "Success"


def test_model_retry_zero_retries() -> None:
    """Test ModelRetryMiddleware with max_retries=0 (no retries)."""
    model = AlwaysFailModel(messages=iter([]))

    retry = ModelRetryMiddleware(
        max_retries=0,  # No retries
        initial_delay=0.01,
        jitter=False,
    )

    agent = create_agent(
        model=model,
        middleware=[retry],
        checkpointer=InMemorySaver(),
    )

    # Should fail immediately without retrying
    with pytest.raises(ValueError, match="Always fails"):
        agent.invoke(
            {"messages": [HumanMessage("Test")]},
            {"configurable": {"thread_id": "test"}},
        )
