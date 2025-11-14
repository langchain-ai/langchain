"""Tests for ModelRetryMiddleware functionality."""

import asyncio
import time
from typing import cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain.agents.factory import create_agent
from langchain.agents.middleware.model_retry import ModelRetryMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langgraph.runtime import Runtime


def _fake_runtime() -> Runtime:
    return cast(Runtime, object())


def _make_request() -> ModelRequest:
    """Create a minimal ModelRequest for testing."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="success")]))
    return ModelRequest(
        model=model,
        system_prompt=None,
        messages=[],
        tool_choice=None,
        tools=[],
        response_format=None,
        state=cast("AgentState", {}),  # type: ignore[name-defined]
        runtime=_fake_runtime(),
        model_settings={},
    )


class TemporaryFailureModel:
    """Model that fails a certain number of times before succeeding."""

    def __init__(self, fail_count: int):
        """Initialize with the number of times to fail.

        Args:
            fail_count: Number of times to fail before succeeding.
        """
        self.fail_count = fail_count
        self.attempt = 0

    def __call__(self) -> AIMessage:
        """Execute the model call.

        Returns:
            Success message if attempt >= fail_count.

        Raises:
            ValueError: If attempt < fail_count.
        """
        self.attempt += 1
        if self.attempt <= self.fail_count:
            msg = f"Temporary failure {self.attempt}"
            raise ValueError(msg)
        return AIMessage(content=f"Success after {self.attempt} attempts")


def test_model_retry_initialization_defaults() -> None:
    """Test ModelRetryMiddleware initialization with default values."""
    retry = ModelRetryMiddleware()

    assert retry.max_retries == 2
    assert retry.tools == []
    assert retry.on_failure == "raise"
    assert retry.backoff_factor == 2.0
    assert retry.initial_delay == 1.0
    assert retry.max_delay == 60.0
    assert retry.jitter is True


def test_model_retry_initialization_custom() -> None:
    """Test ModelRetryMiddleware initialization with custom values."""
    retry = ModelRetryMiddleware(
        max_retries=5,
        retry_on=(ValueError, RuntimeError),
        on_failure="raise",
        backoff_factor=1.5,
        initial_delay=0.5,
        max_delay=30.0,
        jitter=False,
    )

    assert retry.max_retries == 5
    assert retry.tools == []
    assert retry.retry_on == (ValueError, RuntimeError)
    assert retry.on_failure == "raise"
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
    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0.01, jitter=False)
    request = _make_request()

    def mock_handler(req: ModelRequest) -> ModelResponse:
        result = AIMessage(content="success")
        return ModelResponse(result=[result])

    response = retry.wrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "success"


def test_model_retry_failing_model_raises() -> None:
    """Test ModelRetryMiddleware with failing model raises exception."""

    class FailingModel(GenericFakeChatModel):
        def _generate(self, messages, **kwargs):
            raise ValueError("Model failed")

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        on_failure="raise",
    )
    request = _make_request()
    request.model = FailingModel(messages=iter([]))

    def mock_handler(req: ModelRequest) -> ModelResponse:
        result = req.model.invoke([])
        return ModelResponse(result=[result])

    # Should raise the ValueError from the model after all retries
    with pytest.raises(ValueError, match="Model failed"):
        retry.wrap_model_call(request, mock_handler)


def test_model_retry_succeeds_after_retries() -> None:
    """Test ModelRetryMiddleware succeeds after temporary failures."""
    temp_fail = TemporaryFailureModel(fail_count=2)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.01,
        jitter=False,
    )
    request = _make_request()

    def mock_handler(req: ModelRequest) -> ModelResponse:
        result = temp_fail()
        return ModelResponse(result=[result])

    response = retry.wrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert "Success after 3 attempts" in response.result[0].content


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

    # Test that ValueError is retried (should see delay)
    request = _make_request()
    request.model = ValueErrorModel(messages=iter([]))

    def mock_handler(req: ModelRequest) -> ModelResponse:
        result = req.model.invoke([])
        return ModelResponse(result=[result])

    start_time = time.time()
    with pytest.raises(ValueError, match="ValueError occurred"):
        retry.wrap_model_call(request, mock_handler)
    elapsed = time.time() - start_time

    # Should have retried with delays (at least 2 retries with 0.01s delay each)
    assert elapsed >= 0.02

    # Test that RuntimeError is not retried (should fail immediately)
    request.model = RuntimeErrorModel(messages=iter([]))
    start_time = time.time()
    with pytest.raises(RuntimeError, match="RuntimeError occurred"):
        retry.wrap_model_call(request, mock_handler)
    elapsed = time.time() - start_time

    # Should fail immediately without retries
    assert elapsed < 0.01


def test_model_retry_custom_exception_filter() -> None:
    """Test ModelRetryMiddleware with custom exception filter function."""

    class CustomError(Exception):
        """Custom exception with retry_me attribute."""

        def __init__(self, message: str, retry_me: bool):
            """Initialize custom error.

            Args:
                message: Error message.
                retry_me: Whether this error should be retried.
            """
            super().__init__(message)
            self.retry_me = retry_me

    attempt_count = {"value": 0}

    def should_retry(exc: Exception) -> bool:
        return isinstance(exc, CustomError) and exc.retry_me

    retry = ModelRetryMiddleware(
        max_retries=3,
        retry_on=should_retry,
        initial_delay=0.01,
        jitter=False,
    )
    request = _make_request()

    def mock_handler(req: ModelRequest) -> ModelResponse:
        attempt_count["value"] += 1
        if attempt_count["value"] == 1:
            raise CustomError("Retryable error", retry_me=True)
        raise CustomError("Non-retryable error", retry_me=False)

    with pytest.raises(CustomError, match="Non-retryable error"):
        retry.wrap_model_call(request, mock_handler)

    # Should retry once (attempt 1 with retry_me=True), then fail on attempt 2 (retry_me=False)
    assert attempt_count["value"] == 2


def test_model_retry_backoff_timing() -> None:
    """Test ModelRetryMiddleware applies correct backoff delays."""
    temp_fail = TemporaryFailureModel(fail_count=3)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        jitter=False,
    )
    request = _make_request()

    def mock_handler(req: ModelRequest) -> ModelResponse:
        result = temp_fail()
        return ModelResponse(result=[result])

    start_time = time.time()
    response = retry.wrap_model_call(request, mock_handler)
    elapsed = time.time() - start_time

    assert isinstance(response, ModelResponse)
    # Expected delays: 0.1 + 0.2 + 0.4 = 0.7 seconds
    # Allow some margin for execution time
    assert elapsed >= 0.6, f"Expected at least 0.6s, got {elapsed}s"


def test_model_retry_constant_backoff() -> None:
    """Test ModelRetryMiddleware with constant backoff (backoff_factor=0)."""
    temp_fail = TemporaryFailureModel(fail_count=2)

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.1,
        backoff_factor=0.0,  # Constant backoff
        jitter=False,
    )
    request = _make_request()

    def mock_handler(req: ModelRequest) -> ModelResponse:
        result = temp_fail()
        return ModelResponse(result=[result])

    start_time = time.time()
    response = retry.wrap_model_call(request, mock_handler)
    elapsed = time.time() - start_time

    assert isinstance(response, ModelResponse)
    # Expected delays: 0.1 + 0.1 = 0.2 seconds (constant)
    assert elapsed >= 0.15, f"Expected at least 0.15s, got {elapsed}s"
    assert elapsed < 0.5, f"Expected less than 0.5s (exponential would be longer), got {elapsed}s"


def test_model_retry_max_delay_cap() -> None:
    """Test ModelRetryMiddleware caps delay at max_delay."""
    retry = ModelRetryMiddleware(
        max_retries=5,
        initial_delay=1.0,
        backoff_factor=10.0,  # Very aggressive backoff
        max_delay=2.0,  # Cap at 2 seconds
        jitter=False,
    )

    # Test delay calculation
    delay_0 = retry._calculate_delay(0)  # 1.0
    delay_1 = retry._calculate_delay(1)  # 10.0 -> capped to 2.0
    delay_2 = retry._calculate_delay(2)  # 100.0 -> capped to 2.0

    assert delay_0 == 1.0
    assert delay_1 == 2.0
    assert delay_2 == 2.0


def test_model_retry_jitter_variation() -> None:
    """Test ModelRetryMiddleware adds jitter to delays."""
    retry = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=1.0,
        backoff_factor=1.0,
        jitter=True,
    )

    # Generate multiple delays and ensure they vary
    delays = [retry._calculate_delay(0) for _ in range(10)]

    # All delays should be within ±25% of 1.0 (i.e., between 0.75 and 1.25)
    for delay in delays:
        assert 0.75 <= delay <= 1.25

    # Delays should vary (not all the same)
    assert len(set(delays)) > 1


@pytest.mark.asyncio
async def test_model_retry_async_working_model() -> None:
    """Test ModelRetryMiddleware with async execution and working model."""
    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0.01, jitter=False)
    request = _make_request()

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        result = AIMessage(content="success")
        return ModelResponse(result=[result])

    response = await retry.awrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "success"


@pytest.mark.asyncio
async def test_model_retry_async_failing_model() -> None:
    """Test ModelRetryMiddleware with async execution and failing model."""

    class AsyncFailingModel(GenericFakeChatModel):
        async def _agenerate(self, messages, **kwargs):
            raise ValueError("Model failed")

    retry = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0.01,
        jitter=False,
        on_failure="raise",
    )
    request = _make_request()
    request.model = AsyncFailingModel(messages=iter([]))

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        result = await req.model.ainvoke([])
        return ModelResponse(result=[result])

    with pytest.raises(ValueError, match="Model failed"):
        await retry.awrap_model_call(request, mock_handler)


@pytest.mark.asyncio
async def test_model_retry_async_succeeds_after_retries() -> None:
    """Test ModelRetryMiddleware async execution succeeds after temporary failures."""
    temp_fail = TemporaryFailureModel(fail_count=2)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.01,
        jitter=False,
    )
    request = _make_request()

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        result = temp_fail()
        return ModelResponse(result=[result])

    response = await retry.awrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert "Success after 3 attempts" in response.result[0].content


@pytest.mark.asyncio
async def test_model_retry_async_backoff_timing() -> None:
    """Test ModelRetryMiddleware async applies correct backoff delays."""
    temp_fail = TemporaryFailureModel(fail_count=3)

    retry = ModelRetryMiddleware(
        max_retries=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        jitter=False,
    )
    request = _make_request()

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        result = temp_fail()
        return ModelResponse(result=[result])

    start_time = time.time()
    response = await retry.awrap_model_call(request, mock_handler)
    elapsed = time.time() - start_time

    assert isinstance(response, ModelResponse)
    # Expected delays: 0.1 + 0.2 + 0.4 = 0.7 seconds
    assert elapsed >= 0.6, f"Expected at least 0.6s, got {elapsed}s"


def test_model_retry_zero_retries() -> None:
    """Test ModelRetryMiddleware with max_retries=0 (no retries)."""

    class FailingModel(GenericFakeChatModel):
        def _generate(self, messages, **kwargs):
            raise ValueError("Model failed")

    retry = ModelRetryMiddleware(
        max_retries=0,  # No retries
        on_failure="raise",
    )
    request = _make_request()
    request.model = FailingModel(messages=iter([]))

    def mock_handler(req: ModelRequest) -> ModelResponse:
        result = req.model.invoke([])
        return ModelResponse(result=[result])

    # Should fail immediately without retries
    start_time = time.time()
    with pytest.raises(ValueError, match="Model failed"):
        retry.wrap_model_call(request, mock_handler)
    elapsed = time.time() - start_time

    # Should fail immediately (no delay)
    assert elapsed < 0.1


def test_model_retry_with_agent() -> None:
    """Test ModelRetryMiddleware with agent.invoke."""
    attempt_counter = {"value": 0}

    class FailingModel(BaseChatModel):
        """Model that fails twice then succeeds."""

        def _generate(self, messages, **kwargs):
            attempt_counter["value"] += 1
            if attempt_counter["value"] <= 2:
                raise ValueError(f"Failure {attempt_counter['value']}")
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Success after retries"))]
            )

        @property
        def _llm_type(self):
            return "failing"

    model = FailingModel()
    retry = ModelRetryMiddleware(max_retries=3, initial_delay=0.01, jitter=False)

    agent = create_agent(model=model, middleware=[retry])

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Should have succeeded after retries
    assert len(result["messages"]) == 2
    assert result["messages"][1].content == "Success after retries"


def test_model_retry_exhausted_with_agent() -> None:
    """Test ModelRetryMiddleware with agent.invoke when all retries exhausted."""

    class AlwaysFailingModel(BaseChatModel):
        """Model that always fails."""

        def _generate(self, messages, **kwargs):
            raise ValueError("Model failed")

        @property
        def _llm_type(self):
            return "failing"

    model = AlwaysFailingModel()
    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0.01, jitter=False)

    agent = create_agent(model=model, middleware=[retry])

    # Should fail after exhausting retries
    with pytest.raises(ValueError, match="Model failed"):
        agent.invoke({"messages": [HumanMessage("Test")]})
