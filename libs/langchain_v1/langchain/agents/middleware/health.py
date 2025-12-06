"""Health-aware middleware for agents.

This module provides health monitoring, automatic retries, and fallback
capabilities for tools and models in LangChain agents.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, ToolMessage

from langchain.agents.middleware._retry import (
    calculate_delay,
    validate_retry_params,
)
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.store.base import BaseStore
    from langgraph.types import Command

    from langchain.agents.middleware.types import ModelRequest, ToolCallRequest


# Type alias for telemetry emitter
TelemetryEmitter = Callable[[dict[str, Any]], None]


@dataclass
class HealthRecord:
    """Single health record for a dependency call."""

    ok: bool
    """Whether the call succeeded."""

    lat_ms: float
    """Latency of the call in milliseconds."""

    ts: float
    """Timestamp of the call (epoch seconds)."""


@dataclass
class HealthStats:
    """Rolling window statistics for a single dependency (tool or model).

    Tracks success/failure rates and latency metrics over a configurable window.

    Examples:
        ```python
        stats = HealthStats(window_size=100)
        stats.record(ok=True, lat_ms=50.0)
        stats.record(ok=False, lat_ms=150.0)

        print(stats.error_rate())  # 0.5
        print(stats.p95_latency())  # ~150.0
        ```
    """

    window_size: int = 100
    """Maximum number of records to keep in the rolling window."""

    _records: deque[HealthRecord] = field(default_factory=deque)
    """Rolling window of health records."""

    _consecutive_failures: int = 0
    """Count of consecutive failures (resets on success)."""

    _last_degraded_ts: float | None = None
    """Timestamp when dependency was last marked as degraded."""

    def record(self, *, ok: bool, lat_ms: float, ts: float | None = None) -> None:
        """Record a new health observation.

        Args:
            ok: Whether the call succeeded.
            lat_ms: Latency of the call in milliseconds.
            ts: Timestamp of the call. Defaults to current time.
        """
        if ts is None:
            ts = time.time()

        record = HealthRecord(ok=ok, lat_ms=lat_ms, ts=ts)
        self._records.append(record)

        # Trim to window size
        while len(self._records) > self.window_size:
            self._records.popleft()

        # Track consecutive failures
        if ok:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

    def count(self) -> int:
        """Return the number of records in the window."""
        return len(self._records)

    def error_rate(self) -> float:
        """Calculate the error rate in the current window.

        Returns:
            Error rate as a float between 0.0 and 1.0.
            Returns 0.0 if no records exist.
        """
        if not self._records:
            return 0.0
        failures = sum(1 for r in self._records if not r.ok)
        return failures / len(self._records)

    def p95_latency(self) -> float:
        """Calculate the 95th percentile latency in the current window.

        Returns:
            P95 latency in milliseconds.
            Returns 0.0 if no records exist.
        """
        if not self._records:
            return 0.0
        latencies = sorted(r.lat_ms for r in self._records)
        idx = int(len(latencies) * 0.95)
        idx = min(idx, len(latencies) - 1)  # Handle edge case
        return latencies[idx]

    def consecutive_failures(self) -> int:
        """Return the current count of consecutive failures."""
        return self._consecutive_failures

    def mark_degraded(self, ts: float | None = None) -> None:
        """Mark this dependency as degraded at the given timestamp.

        Args:
            ts: Timestamp when degradation started. Defaults to current time.
        """
        if ts is None:
            ts = time.time()
        self._last_degraded_ts = ts

    def is_in_cooldown(self, cooldown_seconds: float) -> bool:
        """Check if the dependency is still in cooldown period.

        Args:
            cooldown_seconds: Duration of cooldown period in seconds.

        Returns:
            True if still in cooldown, False otherwise.
        """
        if self._last_degraded_ts is None:
            return False
        elapsed = time.time() - self._last_degraded_ts
        return elapsed < cooldown_seconds

    def clear_cooldown(self) -> None:
        """Clear the cooldown state, allowing the dependency to be used again."""
        self._last_degraded_ts = None

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of current stats for persistence.

        Returns:
            Dictionary containing current health statistics.
        """
        return {
            "count": self.count(),
            "error_rate": self.error_rate(),
            "p95_ms": self.p95_latency(),
            "consecutive_failures": self._consecutive_failures,
            "last_degraded_ts": self._last_degraded_ts,
        }

    @classmethod
    def restore(cls, data: dict[str, Any], window_size: int = 100) -> HealthStats:
        """Restore HealthStats from a persisted snapshot.

        Note: Individual records are not stored, so after restore the stats
        will have the saved metadata but an empty record window. The stats
        will rebuild as new calls are recorded.

        Args:
            data: Snapshot dictionary from `snapshot()` method.
            window_size: Size of the rolling window.

        Returns:
            Restored HealthStats instance.
        """
        stats = cls(window_size=window_size)
        stats._consecutive_failures = data.get("consecutive_failures", 0)
        stats._last_degraded_ts = data.get("last_degraded_ts")
        return stats


@dataclass
class HealthPolicy:
    """Policy configuration for determining health status.

    Defines thresholds for when a dependency should be considered degraded.

    Examples:
        ```python
        policy = HealthPolicy(
            max_error_rate=0.2,
            consecutive_failures=3,
            latency_p95_ms=800,
            min_samples=20,
            cooldown_seconds=60,
        )

        if not policy.is_healthy(stats):
            print("Dependency is degraded!")
        ```
    """

    max_error_rate: float = 0.2
    """Maximum error rate before marking as degraded (0.0 to 1.0)."""

    consecutive_failures: int = 3
    """Number of consecutive failures to trigger degradation."""

    latency_p95_ms: float | None = None
    """Optional P95 latency threshold in ms. If exceeded, marks as degraded."""

    min_samples: int = 10
    """Minimum number of samples required before applying thresholds."""

    cooldown_seconds: float = 60.0
    """Seconds to wait before retrying a degraded dependency."""

    def is_healthy(self, stats: HealthStats) -> bool:
        """Check if the dependency is healthy based on current stats.

        Args:
            stats: Current health statistics for the dependency.

        Returns:
            True if healthy, False if degraded.
        """
        # Not enough samples yet - assume healthy
        if stats.count() < self.min_samples:
            return True

        # Check consecutive failures
        if stats.consecutive_failures() >= self.consecutive_failures:
            return False

        # Check error rate
        if stats.error_rate() >= self.max_error_rate:
            return False

        # Check latency if threshold is set
        return not (self.latency_p95_ms is not None and stats.p95_latency() >= self.latency_p95_ms)


def _noop_emitter(event: dict[str, Any]) -> None:
    """Default no-op telemetry emitter."""


def create_langsmith_emitter(
    run_name: str = "health_middleware",
    project_name: str | None = None,
) -> TelemetryEmitter:
    """Create a telemetry emitter that logs events to LangSmith.

    Events are logged as custom metadata on a trace run.

    Args:
        run_name: Name for the trace run.
        project_name: Optional LangSmith project name.

    Returns:
        Telemetry emitter function compatible with HealthMiddleware.

    Examples:
        ```python
        from langchain.agents.middleware.health import (
            HealthMiddleware,
            create_langsmith_emitter,
        )

        health = HealthMiddleware(
            emitter=create_langsmith_emitter(project_name="my-project"),
        )
        ```
    """
    try:
        from langsmith import Client
    except ImportError:
        msg = "langsmith is required for LangSmith emitter. Install with: pip install langsmith"
        raise ImportError(msg) from None

    client = Client()

    def emitter(event: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            # Log as a custom run with event as inputs
            client.create_run(
                name=run_name,
                run_type="chain",
                project_name=project_name,
                inputs=event,
            )

    return emitter


def _is_provider_dict_tool(tool: Any) -> bool:
    """Check if a tool is a provider dict (not a BaseTool instance)."""
    return isinstance(tool, dict)


class HealthMiddleware(AgentMiddleware):
    """Health-aware middleware with auto retry and fallback capabilities.

    Monitors per-tool and per-model health statistics, automatically retries
    failed calls with exponential backoff, and supports fallback models and
    tool disabling when dependencies become degraded.

    Examples:
        !!! example "Basic usage with default settings"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import HealthMiddleware, HealthPolicy

            health = HealthMiddleware(
                policy=HealthPolicy(max_error_rate=0.2, consecutive_failures=3),
                fallback_model="openai:gpt-4o-mini",
            )

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[search_tool],
                middleware=[health],
            )
            ```

        !!! example "With telemetry emitter"

            ```python
            def my_emitter(event: dict) -> None:
                print(f"Health event: {event}")


            health = HealthMiddleware(
                policy=HealthPolicy(max_error_rate=0.3),
                emitter=my_emitter,
            )
            ```

        !!! example "Disable flaky tools during run"

            ```python
            health = HealthMiddleware(
                policy=HealthPolicy(consecutive_failures=5),
                disable_tools=True,  # Skip degraded tools
            )
            ```
    """

    def __init__(
        self,
        *,
        policy: HealthPolicy | None = None,
        retry_on_error: int = 0,
        fallback_model: str | BaseChatModel | None = None,
        disable_tools: bool = False,
        preserve_provider_tools: bool = True,
        emitter: TelemetryEmitter | None = None,
        store: BaseStore | None = None,
        window_size: int = 100,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        """Initialize HealthMiddleware.

        Args:
            policy: Health policy configuration. Defaults to sensible defaults.
            retry_on_error: Number of retry attempts on error (0 = no retries).
            fallback_model: Fallback model to use when primary is degraded.
                Can be a string (e.g., "openai:gpt-4o-mini") or BaseChatModel.
            disable_tools: If True, skip degraded tools for the current run.
            preserve_provider_tools: If True, provider dict tools are never disabled
                even when degraded (default True).
            emitter: Callable to receive telemetry events. Defaults to no-op.
            store: Optional BaseStore for persisting health stats across runs.
                Stats are saved under keys like "health:model:<name>".
            window_size: Size of the rolling window for health stats.
            initial_delay: Initial delay in seconds before first retry.
            backoff_factor: Multiplier for exponential backoff.
            max_delay: Maximum delay in seconds between retries.
            jitter: Whether to add random jitter to retry delays.
        """
        super().__init__()

        self.policy = policy or HealthPolicy()
        self.retry_on_error = retry_on_error
        self.disable_tools = disable_tools
        self.preserve_provider_tools = preserve_provider_tools
        self.emitter = emitter or _noop_emitter
        self._store = store
        self.window_size = window_size

        # Retry parameters
        validate_retry_params(retry_on_error, initial_delay, max_delay, backoff_factor)
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter

        # Initialize fallback model
        self._fallback_model: BaseChatModel | None = None
        if fallback_model is not None:
            if isinstance(fallback_model, str):
                self._fallback_model = init_chat_model(fallback_model)
            else:
                self._fallback_model = fallback_model

        # Per-key health stats
        self._model_stats: dict[str, HealthStats] = {}
        self._tool_stats: dict[str, HealthStats] = {}

        # Load persisted stats if store is available
        if self._store is not None:
            self._load_stats_from_store()

        # Required by AgentMiddleware
        self.tools = []

    def _get_model_stats(self, model_name: str) -> HealthStats:
        """Get or create HealthStats for a model."""
        if model_name not in self._model_stats:
            self._model_stats[model_name] = HealthStats(window_size=self.window_size)
        return self._model_stats[model_name]

    def _get_tool_stats(self, tool_name: str) -> HealthStats:
        """Get or create HealthStats for a tool."""
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = HealthStats(window_size=self.window_size)
        return self._tool_stats[tool_name]

    def _emit_event(
        self,
        kind: str,
        name: str,
        *,
        ok: bool,
        lat_ms: float,
        stats: HealthStats,
        is_healthy: bool,
    ) -> None:
        """Emit a telemetry event."""
        event = {
            "kind": kind,
            "name": name,
            "ok": ok,
            "lat_ms": lat_ms,
            "status": "healthy" if is_healthy else "degraded",
            "error_rate": stats.error_rate(),
            "p95_ms": stats.p95_latency(),
            "count": stats.count(),
            "ts": time.time(),
        }
        self.emitter(event)

    def _load_stats_from_store(self) -> None:
        """Load health stats from the store if available."""
        if self._store is None:
            return

        try:
            # Load model stats
            for item in self._store.search(("health", "model")):
                if item.value:
                    name = item.key
                    self._model_stats[name] = HealthStats.restore(
                        item.value, window_size=self.window_size
                    )

            # Load tool stats
            for item in self._store.search(("health", "tool")):
                if item.value:
                    name = item.key
                    self._tool_stats[name] = HealthStats.restore(
                        item.value, window_size=self.window_size
                    )
        except Exception:  # noqa: BLE001, S110
            # Ignore errors loading from store - start fresh
            pass

    def save_stats(self) -> None:
        """Manually save current health stats to the store.

        Call this periodically if you want to persist stats during a run.
        Stats are also saved automatically when getting stats via
        `get_model_stats` or `get_tool_stats` if a store is configured.

        Raises:
            RuntimeError: If no store is configured.
        """
        if self._store is None:
            msg = "No store configured for persistence"
            raise RuntimeError(msg)

        # Save model stats
        for name, stats in self._model_stats.items():
            self._store.put(("health", "model"), name, stats.snapshot())

        # Save tool stats
        for name, stats in self._tool_stats.items():
            self._store.put(("health", "tool"), name, stats.snapshot())

    def _get_model_name(self, request: ModelRequest) -> str:
        """Extract a name identifier for the model."""
        model = request.model
        # Try common attributes for model identification
        if hasattr(model, "model_name"):
            return str(model.model_name)
        if hasattr(model, "model"):
            return str(model.model)
        return model.__class__.__name__

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Intercept model calls with health monitoring and fallback.

        Args:
            request: Model request to execute.
            handler: Callback to execute the model.

        Returns:
            ModelResponse or AIMessage from successful call.
        """
        model_name = self._get_model_name(request)
        stats = self._get_model_stats(model_name)

        # Check if model is degraded and we should use fallback
        is_healthy = self.policy.is_healthy(stats)

        if not is_healthy and self._fallback_model is not None:
            if not stats.is_in_cooldown(self.policy.cooldown_seconds):
                stats.mark_degraded()
            request = request.override(model=self._fallback_model)

        # Attempt with retries
        last_exception: Exception | None = None
        for attempt in range(self.retry_on_error + 1):
            start_time = time.time()
            try:
                response = handler(request)
            except Exception as exc:  # noqa: BLE001
                lat_ms = (time.time() - start_time) * 1000
                stats.record(ok=False, lat_ms=lat_ms)
                is_healthy_after = self.policy.is_healthy(stats)
                self._emit_event(
                    "model",
                    model_name,
                    ok=False,
                    lat_ms=lat_ms,
                    stats=stats,
                    is_healthy=is_healthy_after,
                )
                last_exception = exc

                # Check if we should retry
                if attempt < self.retry_on_error:
                    delay = calculate_delay(
                        attempt,
                        backoff_factor=self.backoff_factor,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        jitter=self.jitter,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    continue
            else:
                lat_ms = (time.time() - start_time) * 1000
                stats.record(ok=True, lat_ms=lat_ms)
                is_healthy_after = self.policy.is_healthy(stats)
                self._emit_event(
                    "model",
                    model_name,
                    ok=True,
                    lat_ms=lat_ms,
                    stats=stats,
                    is_healthy=is_healthy_after,
                )
                return response

        # All retries exhausted - raise last exception
        if last_exception is not None:
            raise last_exception

        # Unreachable
        msg = "Unexpected: model call loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        """Async intercept model calls with health monitoring and fallback.

        Args:
            request: Model request to execute.
            handler: Async callback to execute the model.

        Returns:
            ModelResponse or AIMessage from successful call.
        """
        model_name = self._get_model_name(request)
        stats = self._get_model_stats(model_name)

        # Check if model is degraded and we should use fallback
        is_healthy = self.policy.is_healthy(stats)

        if not is_healthy and self._fallback_model is not None:
            if not stats.is_in_cooldown(self.policy.cooldown_seconds):
                stats.mark_degraded()
            request = request.override(model=self._fallback_model)

        # Attempt with retries
        last_exception: Exception | None = None
        for attempt in range(self.retry_on_error + 1):
            start_time = time.time()
            try:
                response = await handler(request)
            except Exception as exc:  # noqa: BLE001
                lat_ms = (time.time() - start_time) * 1000
                stats.record(ok=False, lat_ms=lat_ms)
                is_healthy_after = self.policy.is_healthy(stats)
                self._emit_event(
                    "model",
                    model_name,
                    ok=False,
                    lat_ms=lat_ms,
                    stats=stats,
                    is_healthy=is_healthy_after,
                )
                last_exception = exc

                # Check if we should retry
                if attempt < self.retry_on_error:
                    delay = calculate_delay(
                        attempt,
                        backoff_factor=self.backoff_factor,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        jitter=self.jitter,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
            else:
                lat_ms = (time.time() - start_time) * 1000
                stats.record(ok=True, lat_ms=lat_ms)
                is_healthy_after = self.policy.is_healthy(stats)
                self._emit_event(
                    "model",
                    model_name,
                    ok=True,
                    lat_ms=lat_ms,
                    stats=stats,
                    is_healthy=is_healthy_after,
                )
                return response

        # All retries exhausted - raise last exception
        if last_exception is not None:
            raise last_exception

        # Unreachable
        msg = "Unexpected: model call loop completed without returning"
        raise RuntimeError(msg)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool calls with health monitoring.

        Args:
            request: Tool call request to execute.
            handler: Callback to execute the tool.

        Returns:
            ToolMessage or Command from the tool execution.
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]
        tool_call_id = request.tool_call.get("id")
        stats = self._get_tool_stats(tool_name)

        # Check if tool is degraded
        is_healthy = self.policy.is_healthy(stats)

        # Skip disabling if this is a provider dict tool and preserve_provider_tools is True
        is_provider_tool = _is_provider_dict_tool(request.tool)
        should_disable = (
            not is_healthy
            and self.disable_tools
            and not (self.preserve_provider_tools and is_provider_tool)
        )

        if should_disable:
            if not stats.is_in_cooldown(self.policy.cooldown_seconds):
                stats.mark_degraded()
            # Return error message instead of executing
            return ToolMessage(
                content=f"Tool '{tool_name}' is currently unavailable due to degraded health.",
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
            )

        # Attempt with retries
        last_exception: Exception | None = None
        for attempt in range(self.retry_on_error + 1):
            start_time = time.time()
            try:
                result = handler(request)
            except Exception as exc:  # noqa: BLE001
                lat_ms = (time.time() - start_time) * 1000
                stats.record(ok=False, lat_ms=lat_ms)
                is_healthy_after = self.policy.is_healthy(stats)
                self._emit_event(
                    "tool",
                    tool_name,
                    ok=False,
                    lat_ms=lat_ms,
                    stats=stats,
                    is_healthy=is_healthy_after,
                )
                last_exception = exc

                # Check if we should retry
                if attempt < self.retry_on_error:
                    delay = calculate_delay(
                        attempt,
                        backoff_factor=self.backoff_factor,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        jitter=self.jitter,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    continue
            else:
                lat_ms = (time.time() - start_time) * 1000
                # Check if result indicates an error (ToolMessage with status="error")
                ok = not (isinstance(result, ToolMessage) and result.status == "error")
                stats.record(ok=ok, lat_ms=lat_ms)
                is_healthy_after = self.policy.is_healthy(stats)
                self._emit_event(
                    "tool",
                    tool_name,
                    ok=ok,
                    lat_ms=lat_ms,
                    stats=stats,
                    is_healthy=is_healthy_after,
                )
                return result

        # All retries exhausted - raise last exception
        if last_exception is not None:
            raise last_exception

        # Unreachable
        msg = "Unexpected: tool call loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async intercept tool calls with health monitoring.

        Args:
            request: Tool call request to execute.
            handler: Async callback to execute the tool.

        Returns:
            ToolMessage or Command from the tool execution.
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]
        tool_call_id = request.tool_call.get("id")
        stats = self._get_tool_stats(tool_name)

        # Check if tool is degraded
        is_healthy = self.policy.is_healthy(stats)

        # Skip disabling if this is a provider dict tool and preserve_provider_tools is True
        is_provider_tool = _is_provider_dict_tool(request.tool)
        should_disable = (
            not is_healthy
            and self.disable_tools
            and not (self.preserve_provider_tools and is_provider_tool)
        )

        if should_disable:
            if not stats.is_in_cooldown(self.policy.cooldown_seconds):
                stats.mark_degraded()
            # Return error message instead of executing
            return ToolMessage(
                content=f"Tool '{tool_name}' is currently unavailable due to degraded health.",
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
            )

        # Attempt with retries
        last_exception: Exception | None = None
        for attempt in range(self.retry_on_error + 1):
            start_time = time.time()
            try:
                result = await handler(request)
            except Exception as exc:  # noqa: BLE001
                lat_ms = (time.time() - start_time) * 1000
                stats.record(ok=False, lat_ms=lat_ms)
                is_healthy_after = self.policy.is_healthy(stats)
                self._emit_event(
                    "tool",
                    tool_name,
                    ok=False,
                    lat_ms=lat_ms,
                    stats=stats,
                    is_healthy=is_healthy_after,
                )
                last_exception = exc

                # Check if we should retry
                if attempt < self.retry_on_error:
                    delay = calculate_delay(
                        attempt,
                        backoff_factor=self.backoff_factor,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        jitter=self.jitter,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
            else:
                lat_ms = (time.time() - start_time) * 1000
                # Check if result indicates an error
                ok = not (isinstance(result, ToolMessage) and result.status == "error")
                stats.record(ok=ok, lat_ms=lat_ms)
                is_healthy_after = self.policy.is_healthy(stats)
                self._emit_event(
                    "tool",
                    tool_name,
                    ok=ok,
                    lat_ms=lat_ms,
                    stats=stats,
                    is_healthy=is_healthy_after,
                )
                return result

        # All retries exhausted - raise last exception
        if last_exception is not None:
            raise last_exception

        # Unreachable
        msg = "Unexpected: tool call loop completed without returning"
        raise RuntimeError(msg)

    def get_model_stats(self, model_name: str) -> HealthStats | None:
        """Get health stats for a specific model.

        Args:
            model_name: Name of the model.

        Returns:
            HealthStats if available, None otherwise.
        """
        return self._model_stats.get(model_name)

    def get_tool_stats(self, tool_name: str) -> HealthStats | None:
        """Get health stats for a specific tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            HealthStats if available, None otherwise.
        """
        return self._tool_stats.get(tool_name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get snapshots of all health stats.

        Returns:
            Dictionary with model and tool stats snapshots.
        """
        return {
            "models": {name: stats.snapshot() for name, stats in self._model_stats.items()},
            "tools": {name: stats.snapshot() for name, stats in self._tool_stats.items()},
        }
