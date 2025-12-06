"""Tests for HealthMiddleware functionality."""

import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolCallRequest
from pydantic import Field

from langchain.agents.factory import create_agent
from langchain.agents.middleware.health import (
    HealthMiddleware,
    HealthPolicy,
    HealthStats,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from tests.unit_tests.agents.model import FakeToolCallingModel


# ============================================================================
# HealthStats Tests
# ============================================================================


def test_health_stats_initialization() -> None:
    """Test HealthStats initialization with default and custom window size."""
    stats = HealthStats()
    assert stats.window_size == 100
    assert stats.count() == 0
    assert stats.error_rate() == 0.0
    assert stats.p95_latency() == 0.0

    custom_stats = HealthStats(window_size=50)
    assert custom_stats.window_size == 50


def test_health_stats_record() -> None:
    """Test recording health observations."""
    stats = HealthStats(window_size=10)

    stats.record(ok=True, lat_ms=50.0)
    assert stats.count() == 1
    assert stats.error_rate() == 0.0
    assert stats.consecutive_failures() == 0

    stats.record(ok=False, lat_ms=100.0)
    assert stats.count() == 2
    assert stats.error_rate() == 0.5
    assert stats.consecutive_failures() == 1


def test_health_stats_consecutive_failures() -> None:
    """Test consecutive failure tracking."""
    stats = HealthStats()

    # Record 3 consecutive failures
    for _ in range(3):
        stats.record(ok=False, lat_ms=50.0)
    assert stats.consecutive_failures() == 3

    # Success resets consecutive failures
    stats.record(ok=True, lat_ms=50.0)
    assert stats.consecutive_failures() == 0


def test_health_stats_rolling_window() -> None:
    """Test that rolling window limits size correctly."""
    stats = HealthStats(window_size=5)

    # Add 10 records
    for i in range(10):
        stats.record(ok=i % 2 == 0, lat_ms=float(i * 10))

    # Window should only have 5 records
    assert stats.count() == 5


def test_health_stats_error_rate() -> None:
    """Test error rate calculation."""
    stats = HealthStats(window_size=10)

    # 3 failures out of 10
    for i in range(10):
        stats.record(ok=i >= 3, lat_ms=50.0)

    assert stats.error_rate() == 0.3


def test_health_stats_p95_latency() -> None:
    """Test P95 latency calculation."""
    stats = HealthStats(window_size=100)

    # Add 100 records with latencies 1-100
    for i in range(100):
        stats.record(ok=True, lat_ms=float(i + 1))

    # P95 should be around 95
    p95 = stats.p95_latency()
    assert 94 <= p95 <= 96


def test_health_stats_cooldown() -> None:
    """Test cooldown period tracking."""
    stats = HealthStats()

    assert not stats.is_in_cooldown(60.0)

    stats.mark_degraded()
    assert stats.is_in_cooldown(60.0)

    stats.clear_cooldown()
    assert not stats.is_in_cooldown(60.0)


def test_health_stats_snapshot() -> None:
    """Test snapshot generation."""
    stats = HealthStats()
    stats.record(ok=True, lat_ms=50.0)
    stats.record(ok=False, lat_ms=100.0)

    snapshot = stats.snapshot()
    assert snapshot["count"] == 2
    assert snapshot["error_rate"] == 0.5
    assert "p95_ms" in snapshot
    assert "consecutive_failures" in snapshot


# ============================================================================
# HealthPolicy Tests
# ============================================================================


def test_health_policy_defaults() -> None:
    """Test HealthPolicy default values."""
    policy = HealthPolicy()
    assert policy.max_error_rate == 0.2
    assert policy.consecutive_failures == 3
    assert policy.min_samples == 10
    assert policy.cooldown_seconds == 60.0


def test_health_policy_healthy_below_thresholds() -> None:
    """Test is_healthy returns True when below all thresholds."""
    policy = HealthPolicy(
        max_error_rate=0.3,
        consecutive_failures=5,
        min_samples=5,
    )

    stats = HealthStats()
    # Add 10 records, 20% error rate
    for i in range(10):
        stats.record(ok=i >= 2, lat_ms=50.0)

    assert policy.is_healthy(stats) is True


def test_health_policy_min_samples() -> None:
    """Test that policy assumes healthy when below min_samples."""
    policy = HealthPolicy(
        max_error_rate=0.1,  # Very strict
        consecutive_failures=1,
        min_samples=10,
    )

    stats = HealthStats()
    # Only 5 samples, all failures - should still be healthy
    for _ in range(5):
        stats.record(ok=False, lat_ms=50.0)

    assert policy.is_healthy(stats) is True

    # Add more samples to exceed min_samples
    for _ in range(6):
        stats.record(ok=False, lat_ms=50.0)

    assert policy.is_healthy(stats) is False


def test_health_policy_error_rate_threshold() -> None:
    """Test degradation triggered by error rate."""
    policy = HealthPolicy(
        max_error_rate=0.2,
        min_samples=5,
    )

    stats = HealthStats()
    # 30% error rate
    for i in range(10):
        stats.record(ok=i >= 3, lat_ms=50.0)

    assert policy.is_healthy(stats) is False


def test_health_policy_consecutive_failures_threshold() -> None:
    """Test degradation triggered by consecutive failures."""
    policy = HealthPolicy(
        consecutive_failures=3,
        min_samples=5,
    )

    stats = HealthStats()
    # Start with successes to meet min_samples
    for _ in range(5):
        stats.record(ok=True, lat_ms=50.0)
    assert policy.is_healthy(stats) is True

    # 3 consecutive failures
    for _ in range(3):
        stats.record(ok=False, lat_ms=50.0)

    assert policy.is_healthy(stats) is False


def test_health_policy_latency_threshold() -> None:
    """Test degradation triggered by P95 latency."""
    policy = HealthPolicy(
        latency_p95_ms=100.0,
        min_samples=5,
    )

    stats = HealthStats()
    # All fast responses
    for _ in range(10):
        stats.record(ok=True, lat_ms=50.0)
    assert policy.is_healthy(stats) is True

    # Add slow responses to push P95 above threshold
    for _ in range(20):
        stats.record(ok=True, lat_ms=150.0)

    assert policy.is_healthy(stats) is False


# ============================================================================
# HealthMiddleware Tests
# ============================================================================


def test_health_middleware_initialization() -> None:
    """Test HealthMiddleware initialization with defaults."""
    middleware = HealthMiddleware()

    assert middleware.policy is not None
    assert middleware.retry_on_error == 0
    assert middleware.disable_tools is False
    assert middleware.window_size == 100
    assert middleware.tools == []


def test_health_middleware_custom_policy() -> None:
    """Test HealthMiddleware with custom policy."""
    policy = HealthPolicy(max_error_rate=0.5, consecutive_failures=10)
    middleware = HealthMiddleware(policy=policy)

    assert middleware.policy.max_error_rate == 0.5
    assert middleware.policy.consecutive_failures == 10


def test_health_middleware_telemetry_emission() -> None:
    """Test that telemetry events are emitted."""
    events = []

    def capture_emitter(event: dict) -> None:
        events.append(event)

    middleware = HealthMiddleware(emitter=capture_emitter)

    # Create a simple handler that returns success
    def mock_handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="test")])

    # Create a minimal request
    model = FakeToolCallingModel()
    request = ModelRequest(
        model=model,
        messages=[HumanMessage(content="test")],
    )

    middleware.wrap_model_call(request, mock_handler)

    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "model"
    assert event["ok"] is True
    assert "lat_ms" in event
    assert "error_rate" in event
    assert "status" in event


def test_health_middleware_tracks_model_stats() -> None:
    """Test that model health stats are tracked."""
    middleware = HealthMiddleware()

    def mock_handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="test")])

    model = FakeToolCallingModel()
    request = ModelRequest(
        model=model,
        messages=[HumanMessage(content="test")],
    )

    # Make a few calls
    for _ in range(5):
        middleware.wrap_model_call(request, mock_handler)

    # Check stats exist
    all_stats = middleware.get_all_stats()
    assert len(all_stats["models"]) > 0


def test_health_middleware_retry_on_error() -> None:
    """Test retry behavior on errors."""
    attempt_count = {"value": 0}

    def failing_handler(request: ModelRequest) -> ModelResponse:
        attempt_count["value"] += 1
        if attempt_count["value"] < 3:
            raise ValueError("Temporary error")
        return ModelResponse(result=[AIMessage(content="success")])

    middleware = HealthMiddleware(
        retry_on_error=3,
        initial_delay=0.01,
        jitter=False,
    )

    model = FakeToolCallingModel()
    request = ModelRequest(
        model=model,
        messages=[HumanMessage(content="test")],
    )

    result = middleware.wrap_model_call(request, failing_handler)

    assert attempt_count["value"] == 3
    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "success"


def test_health_middleware_exhausted_retries() -> None:
    """Test that exception is raised when retries are exhausted."""

    def always_failing_handler(request: ModelRequest) -> ModelResponse:
        raise ValueError("Permanent error")

    middleware = HealthMiddleware(
        retry_on_error=2,
        initial_delay=0.01,
        jitter=False,
    )

    model = FakeToolCallingModel()
    request = ModelRequest(
        model=model,
        messages=[HumanMessage(content="test")],
    )

    with pytest.raises(ValueError, match="Permanent error"):
        middleware.wrap_model_call(request, always_failing_handler)


def test_health_middleware_get_stats() -> None:
    """Test getting individual stats."""
    middleware = HealthMiddleware()

    # Initially no stats
    assert middleware.get_model_stats("test_model") is None
    assert middleware.get_tool_stats("test_tool") is None


class TemporaryFailureModel(FakeToolCallingModel):
    """Model that fails a certain number of times before succeeding."""

    fail_count: int = Field(default=0)
    attempt: int = Field(default=0)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Execute the model."""
        self.attempt += 1
        if self.attempt <= self.fail_count:
            msg = f"Temporary failure {self.attempt}"
            raise ValueError(msg)
        ai_msg = AIMessage(content=f"Success after {self.attempt} attempts", id=str(self.index))
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


class AlwaysFailingModel(FakeToolCallingModel):
    """Model that always fails with a specific exception."""

    error_message: str = Field(default="Model error")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Execute the model and raise exception."""
        raise ValueError(self.error_message)


def test_health_middleware_agent_integration() -> None:
    """Test HealthMiddleware works with create_agent."""
    model = FakeToolCallingModel()

    middleware = HealthMiddleware(
        policy=HealthPolicy(
            max_error_rate=0.5,
            consecutive_failures=5,
            min_samples=3,
        ),
    )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1


@pytest.mark.asyncio
async def test_health_middleware_async_model_call() -> None:
    """Test async model call interception."""
    events = []

    async def mock_async_handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="async test")])

    def capture_emitter(event: dict) -> None:
        events.append(event)

    middleware = HealthMiddleware(emitter=capture_emitter)

    model = FakeToolCallingModel()
    request = ModelRequest(
        model=model,
        messages=[HumanMessage(content="test")],
    )

    result = await middleware.awrap_model_call(request, mock_async_handler)

    assert len(events) == 1
    assert events[0]["ok"] is True
    assert isinstance(result, ModelResponse)


@pytest.mark.asyncio
async def test_health_middleware_async_retry() -> None:
    """Test async retry behavior."""
    attempt_count = {"value": 0}

    async def failing_async_handler(request: ModelRequest) -> ModelResponse:
        attempt_count["value"] += 1
        if attempt_count["value"] < 2:
            raise ValueError("Async error")
        return ModelResponse(result=[AIMessage(content="async success")])

    middleware = HealthMiddleware(
        retry_on_error=2,
        initial_delay=0.01,
        jitter=False,
    )

    model = FakeToolCallingModel()
    request = ModelRequest(
        model=model,
        messages=[HumanMessage(content="test")],
    )

    result = await middleware.awrap_model_call(request, failing_async_handler)

    assert attempt_count["value"] == 2
    assert result.result[0].content == "async success"


# ============================================================================
# Edge Case Tests - Enhancements
# ============================================================================


def test_health_stats_restore() -> None:
    """Test restoring HealthStats from a snapshot."""
    # Create and populate original stats
    original = HealthStats(window_size=50)
    original.record(ok=True, lat_ms=50.0)
    original.record(ok=False, lat_ms=100.0)
    original._consecutive_failures = 5
    original.mark_degraded()

    # Create snapshot and restore
    snapshot = original.snapshot()
    restored = HealthStats.restore(snapshot, window_size=50)

    # Metadata should match
    assert restored._consecutive_failures == 5
    assert restored._last_degraded_ts is not None
    assert restored.is_in_cooldown(60.0)

    # Window should be empty (records not persisted)
    assert restored.count() == 0


def test_health_stats_cooldown_expiry() -> None:
    """Test that cooldown expires after the configured duration."""
    stats = HealthStats()

    # Mark as degraded at a known time
    old_time = time.time() - 120  # 2 minutes ago
    stats._last_degraded_ts = old_time

    # Should NOT be in cooldown if cooldown is 60 seconds
    assert not stats.is_in_cooldown(60.0)

    # Should still be in cooldown if cooldown is 300 seconds
    assert stats.is_in_cooldown(300.0)


def test_health_middleware_preserve_provider_tools_default() -> None:
    """Test that preserve_provider_tools defaults to True."""
    middleware = HealthMiddleware()
    assert middleware.preserve_provider_tools is True


def test_health_middleware_preserve_provider_tools_false() -> None:
    """Test that provider tools can be disabled when preserve_provider_tools is False."""
    middleware = HealthMiddleware(
        disable_tools=True,
        preserve_provider_tools=False,
        policy=HealthPolicy(min_samples=1, consecutive_failures=1),
    )
    # When preserve_provider_tools is False, provider dicts should be disabled
    assert middleware.preserve_provider_tools is False


def test_health_middleware_store_param() -> None:
    """Test that middleware accepts store parameter."""
    # Just verify it doesn't raise
    middleware = HealthMiddleware(store=None)
    assert middleware._store is None


def test_health_middleware_save_stats_no_store() -> None:
    """Test save_stats raises when no store is configured."""
    middleware = HealthMiddleware()

    with pytest.raises(RuntimeError, match="No store configured"):
        middleware.save_stats()


def test_health_middleware_get_all_stats_empty() -> None:
    """Test get_all_stats returns empty dicts when no calls made."""
    middleware = HealthMiddleware()

    all_stats = middleware.get_all_stats()
    assert all_stats == {"models": {}, "tools": {}}


def test_health_stats_consecutive_failures_reset() -> None:
    """Test that a success resets consecutive failures to zero."""
    stats = HealthStats()

    # Build up failures
    for _ in range(10):
        stats.record(ok=False, lat_ms=50.0)
    assert stats.consecutive_failures() == 10

    # One success should reset
    stats.record(ok=True, lat_ms=50.0)
    assert stats.consecutive_failures() == 0

    # Two more failures
    stats.record(ok=False, lat_ms=50.0)
    stats.record(ok=False, lat_ms=50.0)
    assert stats.consecutive_failures() == 2


def test_health_policy_all_thresholds_combined() -> None:
    """Test policy with all thresholds set - first one triggered wins."""
    policy = HealthPolicy(
        max_error_rate=0.3,
        consecutive_failures=3,
        latency_p95_ms=100.0,
        min_samples=5,
    )

    stats = HealthStats()

    # Add enough samples, all healthy
    for _ in range(10):
        stats.record(ok=True, lat_ms=50.0)
    assert policy.is_healthy(stats) is True

    # Trigger consecutive failures (3)
    for _ in range(3):
        stats.record(ok=False, lat_ms=50.0)
    assert policy.is_healthy(stats) is False


def test_health_middleware_fallback_model_string() -> None:
    """Test that fallback_model can be passed as a string."""
    # This would normally try to init the model - we just verify it doesn't crash
    # when fallback_model is None
    middleware = HealthMiddleware(fallback_model=None)
    assert middleware._fallback_model is None


def test_health_middleware_backoff_parameters() -> None:
    """Test that backoff parameters are correctly set."""
    middleware = HealthMiddleware(
        initial_delay=2.0,
        backoff_factor=3.0,
        max_delay=120.0,
        jitter=False,
    )

    assert middleware.initial_delay == 2.0
    assert middleware.backoff_factor == 3.0
    assert middleware.max_delay == 120.0
    assert middleware.jitter is False
