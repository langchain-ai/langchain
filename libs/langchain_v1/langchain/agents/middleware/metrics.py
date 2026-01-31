"""Metrics and telemetry middleware for agents.

This module provides observability for agent execution through metrics collection
and export to various backends including Prometheus, Datadog, and custom callbacks.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    NamedTuple,
    Protocol,
    cast,
    runtime_checkable,
)

from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
    ResponseT,
)

if TYPE_CHECKING:
    from langchain_core.messages import ToolMessage
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from langchain.agents.middleware.types import (
        ModelRequest,
        ModelResponse,
    )


try:
    from opentelemetry import trace as otel_trace
    _HAS_OPENTELEMETRY = True
except ImportError:
    _HAS_OPENTELEMETRY = False
    otel_trace = None  # type: ignore[assignment]


class _TokenUsage(NamedTuple):
    """Token usage information from a model call."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class _TraceContext(NamedTuple):
    """Trace context for correlation with distributed tracing systems."""

    trace_id: str | None = None
    span_id: str | None = None
    run_id: str | None = None


@dataclass
class ModelCallMetrics:
    """Metrics for a single model call.

    Attributes:
        timestamp: When the call was made.
        latency_ms: Call duration in milliseconds.
        model_name: Name of the model called.
        input_tokens: Number of input tokens (if available).
        output_tokens: Number of output tokens (if available).
        total_tokens: Total tokens used (if available).
        error: Exception if the call failed, `None` otherwise.
        trace_id: OpenTelemetry trace ID for distributed tracing correlation.
        span_id: OpenTelemetry span ID for distributed tracing correlation.
        run_id: LangSmith run ID for trace correlation.
    """

    timestamp: datetime
    latency_ms: float
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    error: BaseException | None = None
    trace_id: str | None = None
    span_id: str | None = None
    run_id: str | None = None

    @property
    def success(self) -> bool:
        """Whether the call succeeded."""
        return self.error is None


@dataclass
class ToolCallMetrics:
    """Metrics for a single tool call.

    Attributes:
        timestamp: When the call was made.
        tool_name: Name of the tool called.
        latency_ms: Call duration in milliseconds.
        error: Exception if the call failed, `None` otherwise.
        trace_id: OpenTelemetry trace ID for distributed tracing correlation.
        span_id: OpenTelemetry span ID for distributed tracing correlation.
        run_id: LangSmith run ID for trace correlation.
    """

    timestamp: datetime
    tool_name: str
    latency_ms: float
    error: BaseException | None = None
    trace_id: str | None = None
    span_id: str | None = None
    run_id: str | None = None

    @property
    def success(self) -> bool:
        """Whether the call succeeded."""
        return self.error is None


@dataclass
class AgentRunMetrics:
    """Aggregated metrics for an agent run.

    Attributes:
        run_id: Unique identifier for this run.
        start_time: When the run started.
        end_time: When the run completed.
        model_calls: List of model call metrics.
        tool_calls: List of tool call metrics.
    """

    run_id: str
    start_time: datetime
    end_time: datetime | None = None
    model_calls: list[ModelCallMetrics] = field(default_factory=list)
    tool_calls: list[ToolCallMetrics] = field(default_factory=list)

    @property
    def total_latency_ms(self) -> float:
        """Total latency across all model and tool calls."""
        model_latency = sum(m.latency_ms for m in self.model_calls)
        tool_latency = sum(t.latency_ms for t in self.tool_calls)
        return model_latency + tool_latency

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all model calls."""
        return sum(m.total_tokens or 0 for m in self.model_calls)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all model calls."""
        return sum(m.input_tokens or 0 for m in self.model_calls)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all model calls."""
        return sum(m.output_tokens or 0 for m in self.model_calls)

    @property
    def model_call_count(self) -> int:
        """Number of model calls in this run."""
        return len(self.model_calls)

    @property
    def tool_call_count(self) -> int:
        """Number of tool calls in this run."""
        return len(self.tool_calls)

    @property
    def tool_success_rate(self) -> float:
        """Tool call success rate (0.0-1.0)."""
        if not self.tool_calls:
            return 1.0
        return sum(1 for t in self.tool_calls if t.success) / len(self.tool_calls)

    @property
    def model_success_rate(self) -> float:
        """Model call success rate (0.0-1.0)."""
        if not self.model_calls:
            return 1.0
        return sum(1 for m in self.model_calls if m.success) / len(self.model_calls)


@runtime_checkable
class MetricsExporter(Protocol):
    """Protocol for metrics exporters.

    Implement this protocol to send metrics to your observability backend.
    Both sync and async methods are provided. Sync methods are called from
    `agent.invoke()`, async methods from `agent.ainvoke()`.

    Example:
        ```python
        class MyCustomExporter:
            def export_model_call(self, metrics: ModelCallMetrics) -> None:
                save_to_database_sync(metrics)

            def export_tool_call(self, metrics: ToolCallMetrics) -> None:
                save_to_database_sync(metrics)

            def export_run_complete(self, metrics: AgentRunMetrics) -> None:
                save_to_database_sync(metrics)

            async def aexport_model_call(self, metrics: ModelCallMetrics) -> None:
                await save_to_database(metrics)

            async def aexport_tool_call(self, metrics: ToolCallMetrics) -> None:
                await save_to_database(metrics)

            async def aexport_run_complete(self, metrics: AgentRunMetrics) -> None:
                await save_to_database(metrics)
        ```
    """

    def export_model_call(self, metrics: ModelCallMetrics) -> None:
        """Export metrics for a single model call (sync)."""
        ...

    def export_tool_call(self, metrics: ToolCallMetrics) -> None:
        """Export metrics for a single tool call (sync)."""
        ...

    def export_run_complete(self, metrics: AgentRunMetrics) -> None:
        """Export aggregated metrics when a run completes (sync)."""
        ...

    async def aexport_model_call(self, metrics: ModelCallMetrics) -> None:
        """Export metrics for a single model call (async)."""
        ...

    async def aexport_tool_call(self, metrics: ToolCallMetrics) -> None:
        """Export metrics for a single tool call (async)."""
        ...

    async def aexport_run_complete(self, metrics: AgentRunMetrics) -> None:
        """Export aggregated metrics when a run completes (async)."""
        ...


class CallbackMetricsExporter:
    """Exporter that invokes callbacks for each metric event.

    Simple and flexible exporter for custom metric handling without
    external dependencies. Supports both sync and async callbacks.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import MetricsMiddleware, CallbackMetricsExporter


        def on_model_call(metrics: ModelCallMetrics) -> None:
            print(f"Model call: {metrics.latency_ms:.1f}ms, {metrics.total_tokens} tokens")


        async def aon_model_call(metrics: ModelCallMetrics) -> None:
            await save_to_db(metrics)


        exporter = CallbackMetricsExporter(
            on_model_call=on_model_call,
            aon_model_call=aon_model_call,
        )
        agent = create_agent("openai:gpt-4o", middleware=[MetricsMiddleware(exporter=exporter)])
        ```
    """

    def __init__(
        self,
        *,
        on_model_call: Callable[[ModelCallMetrics], None] | None = None,
        on_tool_call: Callable[[ToolCallMetrics], None] | None = None,
        on_run_complete: Callable[[AgentRunMetrics], None] | None = None,
        aon_model_call: Callable[[ModelCallMetrics], Awaitable[None]] | None = None,
        aon_tool_call: Callable[[ToolCallMetrics], Awaitable[None]] | None = None,
        aon_run_complete: Callable[[AgentRunMetrics], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize the callback exporter.

        Args:
            on_model_call: Sync callback invoked for each model call.
            on_tool_call: Sync callback invoked for each tool call.
            on_run_complete: Sync callback invoked when an agent run completes.
            aon_model_call: Async callback invoked for each model call.
            aon_tool_call: Async callback invoked for each tool call.
            aon_run_complete: Async callback invoked when an agent run completes.
        """
        self._on_model_call = on_model_call
        self._on_tool_call = on_tool_call
        self._on_run_complete = on_run_complete
        self._aon_model_call = aon_model_call
        self._aon_tool_call = aon_tool_call
        self._aon_run_complete = aon_run_complete

    def export_model_call(self, metrics: ModelCallMetrics) -> None:
        """Export model call metrics via sync callback."""
        if self._on_model_call:
            self._on_model_call(metrics)

    def export_tool_call(self, metrics: ToolCallMetrics) -> None:
        """Export tool call metrics via sync callback."""
        if self._on_tool_call:
            self._on_tool_call(metrics)

    def export_run_complete(self, metrics: AgentRunMetrics) -> None:
        """Export run metrics via sync callback."""
        if self._on_run_complete:
            self._on_run_complete(metrics)

    async def aexport_model_call(self, metrics: ModelCallMetrics) -> None:
        """Export model call metrics via async callback."""
        if self._aon_model_call:
            await self._aon_model_call(metrics)
        elif self._on_model_call:
            self._on_model_call(metrics)

    async def aexport_tool_call(self, metrics: ToolCallMetrics) -> None:
        """Export tool call metrics via async callback."""
        if self._aon_tool_call:
            await self._aon_tool_call(metrics)
        elif self._on_tool_call:
            self._on_tool_call(metrics)

    async def aexport_run_complete(self, metrics: AgentRunMetrics) -> None:
        """Export run metrics via async callback."""
        if self._aon_run_complete:
            await self._aon_run_complete(metrics)
        elif self._on_run_complete:
            self._on_run_complete(metrics)


class InMemoryMetricsExporter:
    """Exporter that stores metrics in memory for testing and inspection.

    Useful for unit testing, debugging, and simple metric aggregation.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import MetricsMiddleware, InMemoryMetricsExporter

        exporter = InMemoryMetricsExporter()
        agent = create_agent("openai:gpt-4o", middleware=[MetricsMiddleware(exporter=exporter)])

        # Works with both sync and async invocation
        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        # or: result = await agent.ainvoke({"messages": [HumanMessage("Hello")]})

        # Inspect collected metrics
        print(f"Total model calls: {len(exporter.model_calls)}")
        print(f"Total tokens: {sum(m.total_tokens or 0 for m in exporter.model_calls)}")
        print(f"Average latency: {exporter.average_model_latency_ms:.1f}ms")
        ```
    """

    def __init__(self) -> None:
        """Initialize the in-memory exporter."""
        self.model_calls: list[ModelCallMetrics] = []
        self.tool_calls: list[ToolCallMetrics] = []
        self.runs: list[AgentRunMetrics] = []
        self._total_tokens: int = 0
        self._total_model_latency_ms: float = 0.0
        self._total_tool_latency_ms: float = 0.0

    def _store_model_call(self, metrics: ModelCallMetrics) -> None:
        """Internal method to store model call metrics."""
        self.model_calls.append(metrics)
        self._total_tokens += metrics.total_tokens or 0
        self._total_model_latency_ms += metrics.latency_ms

    def _store_tool_call(self, metrics: ToolCallMetrics) -> None:
        """Internal method to store tool call metrics."""
        self.tool_calls.append(metrics)
        self._total_tool_latency_ms += metrics.latency_ms

    def _store_run(self, metrics: AgentRunMetrics) -> None:
        """Internal method to store run metrics."""
        self.runs.append(metrics)

    def export_model_call(self, metrics: ModelCallMetrics) -> None:
        """Store model call metrics (sync)."""
        self._store_model_call(metrics)

    def export_tool_call(self, metrics: ToolCallMetrics) -> None:
        """Store tool call metrics (sync)."""
        self._store_tool_call(metrics)

    def export_run_complete(self, metrics: AgentRunMetrics) -> None:
        """Store run metrics (sync)."""
        self._store_run(metrics)

    async def aexport_model_call(self, metrics: ModelCallMetrics) -> None:
        """Store model call metrics (async)."""
        self._store_model_call(metrics)

    async def aexport_tool_call(self, metrics: ToolCallMetrics) -> None:
        """Store tool call metrics (async)."""
        self._store_tool_call(metrics)

    async def aexport_run_complete(self, metrics: AgentRunMetrics) -> None:
        """Store run metrics (async)."""
        self._store_run(metrics)

    def clear(self) -> None:
        """Clear all stored metrics."""
        self.model_calls.clear()
        self.tool_calls.clear()
        self.runs.clear()
        self._total_tokens = 0
        self._total_model_latency_ms = 0.0
        self._total_tool_latency_ms = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens across all model calls."""
        return self._total_tokens

    @property
    def average_model_latency_ms(self) -> float:
        """Average model call latency in milliseconds."""
        if not self.model_calls:
            return 0.0
        return self._total_model_latency_ms / len(self.model_calls)

    @property
    def average_tool_latency_ms(self) -> float:
        """Average tool call latency in milliseconds."""
        if not self.tool_calls:
            return 0.0
        return self._total_tool_latency_ms / len(self.tool_calls)


class MetricsMultiExporter:
    """Exporter that forwards metrics to multiple exporters.

    Use this to send metrics to multiple backends simultaneously.

    Example:
        ```python
        from langchain.agents.middleware import (
            MetricsMiddleware,
            MetricsMultiExporter,
            InMemoryMetricsExporter,
        )

        # Send to multiple exporters
        memory = InMemoryMetricsExporter()
        custom = MyCustomExporter()
        exporter = MetricsMultiExporter([memory, custom])

        agent = create_agent("openai:gpt-4o", middleware=[MetricsMiddleware(exporter=exporter)])
        ```
    """

    def __init__(self, exporters: list[MetricsExporter]) -> None:
        """Initialize the composite exporter.

        Args:
            exporters: List of exporters to forward metrics to.
        """
        self._exporters = exporters

    def export_model_call(self, metrics: ModelCallMetrics) -> None:
        """Forward model call metrics to all exporters (sync)."""
        for exporter in self._exporters:
            exporter.export_model_call(metrics)

    def export_tool_call(self, metrics: ToolCallMetrics) -> None:
        """Forward tool call metrics to all exporters (sync)."""
        for exporter in self._exporters:
            exporter.export_tool_call(metrics)

    def export_run_complete(self, metrics: AgentRunMetrics) -> None:
        """Forward run metrics to all exporters (sync)."""
        for exporter in self._exporters:
            exporter.export_run_complete(metrics)

    async def aexport_model_call(self, metrics: ModelCallMetrics) -> None:
        """Forward model call metrics to all exporters (async)."""
        for exporter in self._exporters:
            await exporter.aexport_model_call(metrics)

    async def aexport_tool_call(self, metrics: ToolCallMetrics) -> None:
        """Forward tool call metrics to all exporters (async)."""
        for exporter in self._exporters:
            await exporter.aexport_tool_call(metrics)

    async def aexport_run_complete(self, metrics: AgentRunMetrics) -> None:
        """Forward run metrics to all exporters (async)."""
        for exporter in self._exporters:
            await exporter.aexport_run_complete(metrics)


_KEY_RUN_ID = "_metrics_run_id"
_KEY_RUN_START = "_metrics_run_start"
_KEY_MODEL_CALLS = "_metrics_model_calls"
_KEY_TOOL_CALLS = "_metrics_tool_calls"


class MetricsState(AgentState[ResponseT], Generic[ResponseT]):
    """State schema for `MetricsMiddleware`.

    Extends `AgentState` with metrics tracking fields. These fields use
    `UntrackedValue` so they reset between runs and `PrivateStateAttr`
    so they don't appear in input/output schemas.
    """

    _metrics_run_id: NotRequired[Annotated[str | None, UntrackedValue, PrivateStateAttr]]
    _metrics_run_start: NotRequired[Annotated[float | None, UntrackedValue, PrivateStateAttr]]
    _metrics_model_calls: NotRequired[
        Annotated[list[ModelCallMetrics], UntrackedValue, PrivateStateAttr]
    ]
    _metrics_tool_calls: NotRequired[
        Annotated[list[ToolCallMetrics], UntrackedValue, PrivateStateAttr]
    ]


class MetricsMiddleware(AgentMiddleware[MetricsState[ResponseT], Any], Generic[ResponseT]):
    """Collect and export metrics for agent execution.

    This middleware tracks observability metrics for agent execution.
    Metrics can be exported to various backends (e.g., Prometheus, Datadog, or custom callbacks).
    You may build your custom exporter by implementing the `MetricsExporter` protocol.

    Examples:
        !!! example "Basic usage with callback exporter"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import MetricsMiddleware, CallbackMetricsExporter


            def log_model_call(metrics):
                print(f"Model: {metrics.latency_ms:.1f}ms, {metrics.total_tokens} tokens")


            exporter = CallbackMetricsExporter(on_model_call=log_model_call)
            agent = create_agent(
                "openai:gpt-4o",
                middleware=[MetricsMiddleware(exporter=exporter)],
            )
            ```

        !!! example "In-memory metrics for testing"

            ```python
            from langchain.agents.middleware import MetricsMiddleware, InMemoryMetricsExporter

            exporter = InMemoryMetricsExporter()
            agent = create_agent(
                "openai:gpt-4o",
                middleware=[MetricsMiddleware(exporter=exporter)],
            )

            result = agent.invoke({"messages": [HumanMessage("Hello")]})

            # Inspect metrics
            print(f"Total tokens: {exporter.total_tokens}")
            print(f"Avg latency: {exporter.average_model_latency_ms:.1f}ms")
            ```

    """

    state_schema: type[MetricsState[Any]] = cast("type[MetricsState[Any]]", MetricsState)

    def __init__(
        self,
        *,
        exporter: MetricsExporter | None = None,
    ) -> None:
        """Initialize the metrics middleware.

        Args:
            exporter: Backend to export metrics to.

                If `None`, metrics are collected in state but not exported.
                Use with state inspection or add an exporter later.

                Exporters control what metrics to process. To selectively track
                only certain metrics, implement an exporter with no-op methods
                for metrics you want to ignore.
        """
        super().__init__()
        self.exporter = exporter
        self.tools: list[Any] = []

    @override
    def before_agent(
        self, state: MetricsState[ResponseT], runtime: Runtime
    ) -> dict[str, Any] | None:
        """Initialize run metrics at the start of agent execution.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with initialized metrics tracking fields.
        """
        return {
            _KEY_RUN_ID: str(uuid.uuid4()),
            _KEY_RUN_START: time.perf_counter(),
            _KEY_MODEL_CALLS: [],
            _KEY_TOOL_CALLS: [],
        }

    async def abefore_agent(
        self, state: MetricsState[ResponseT], runtime: Runtime
    ) -> dict[str, Any] | None:
        """Async version of `before_agent`.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with initialized metrics tracking fields.
        """
        return self.before_agent(state, runtime)

    def _store_model_metrics(
        self, state: AgentState[Any], metrics: ModelCallMetrics
    ) -> None:
        """Store model call metrics in state.

        Args:
            state: The agent state to store metrics in.
            metrics: The model call metrics to store.
        """
        model_calls = state.get(_KEY_MODEL_CALLS)
        if model_calls is None:
            # State not initialized - before_agent wasn't called.
            # This shouldn't happen in normal operation.
            return
        model_calls.append(metrics)

    def _store_tool_metrics(
        self, state: AgentState[Any], metrics: ToolCallMetrics
    ) -> None:
        """Store tool call metrics in state.

        Args:
            state: The agent state to store metrics in.
            metrics: The tool call metrics to store.
        """
        tool_calls = state.get(_KEY_TOOL_CALLS)
        if tool_calls is None:
            # State not initialized - before_agent wasn't called.
            # This shouldn't happen in normal operation.
            return
        tool_calls.append(metrics)

    def _build_trace_context(self, request: ModelRequest | Any) -> _TraceContext:
        """Extract trace context for correlation with distributed tracing.

        Extracts OpenTelemetry trace/span IDs (if available) and LangSmith
        run ID from the request's config or callbacks.

        Args:
            request: The model or tool request.

        Returns:
            _TraceContext with trace_id, span_id, and run_id.
        """
        trace_id, span_id = self._get_otel_context()
        run_id = self._get_run_id(request)
        return _TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            run_id=run_id,
        )

    def _get_otel_context(self) -> tuple[str | None, str | None]:
        """Extract OpenTelemetry trace and span IDs from current context.

        Returns:
            Tuple of (trace_id, span_id) as hex strings, or (None, None)
            if OpenTelemetry is not installed or no active span.
        """
        if not _HAS_OPENTELEMETRY or otel_trace is None:
            return None, None

        span = otel_trace.get_current_span()
        if span is None:
            return None, None

        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return None, None

        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")

    def _get_run_id(self, request: ModelRequest | Any) -> str | None:
        """Extract LangSmith run ID from request config or callbacks.

        Args:
            request: The model or tool request with optional config.

        Returns:
            LangSmith run ID as string, or None if not available.
        """
        config = getattr(request, "config", None)
        if config is None:
            return None

        if isinstance(config, dict):
            run_id = config.get("run_id")
            if run_id is not None:
                return str(run_id)

            callbacks = config.get("callbacks", [])
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "run_id") and cb.run_id is not None:
                        return str(cb.run_id)
                    if hasattr(cb, "parent_run_id") and cb.parent_run_id is not None:
                        return str(cb.parent_run_id)

        return None

    def _extract_token_usage(self, response: ModelResponse) -> _TokenUsage:
        """Extract token usage from model response.

        Args:
            response: The model response.

        Returns:
            _TokenUsage named tuple with input, output, and total token counts.
        """
        if not response.result:
            return _TokenUsage()

        ai_message = response.result[0]
        if not hasattr(ai_message, "usage_metadata") or not ai_message.usage_metadata:
            return _TokenUsage()

        usage = ai_message.usage_metadata
        return _TokenUsage(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    def _extract_model_name(self, request: ModelRequest) -> str | None:
        """Extract model name from request.

        Args:
            request: The model request.

        Returns:
            Model name or `None` if not available.
        """
        return (
            getattr(request.model, "model_name", None)
            or getattr(request.model, "model", None)
            or getattr(request.model, "model_id", None)
        )

    def _build_model_call_metrics(
        self,
        request: ModelRequest,
        response: ModelResponse | None,
        start_time: float,
        error: BaseException | None,
        trace_ctx: _TraceContext,
    ) -> ModelCallMetrics:
        """Build model call metrics from execution context.

        Args:
            request: The model request.
            response: The model response (or None if failed).
            start_time: The start time from perf_counter.
            error: Exception if the call failed.
            trace_ctx: Trace context for correlation.

        Returns:
            ModelCallMetrics instance.
        """
        latency_ms = (time.perf_counter() - start_time) * 1000
        token_usage = self._extract_token_usage(response) if response else _TokenUsage()

        return ModelCallMetrics(
            timestamp=datetime.now(tz=timezone.utc),
            latency_ms=latency_ms,
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
            total_tokens=token_usage.total_tokens,
            model_name=self._extract_model_name(request),
            error=error,
            trace_id=trace_ctx.trace_id,
            span_id=trace_ctx.span_id,
            run_id=trace_ctx.run_id,
        )

    def _build_tool_call_metrics(
        self,
        tool_name: str,
        start_time: float,
        error: BaseException | None,
        trace_ctx: _TraceContext,
    ) -> ToolCallMetrics:
        """Build tool call metrics from execution context.

        Args:
            tool_name: Name of the tool called.
            start_time: The start time from perf_counter.
            error: Exception if the call failed.
            trace_ctx: Trace context for correlation.

        Returns:
            ToolCallMetrics instance.
        """
        latency_ms = (time.perf_counter() - start_time) * 1000

        return ToolCallMetrics(
            timestamp=datetime.now(tz=timezone.utc),
            tool_name=tool_name,
            latency_ms=latency_ms,
            error=error,
            trace_id=trace_ctx.trace_id,
            span_id=trace_ctx.span_id,
            run_id=trace_ctx.run_id,
        )

    def _build_run_metrics(self, state: MetricsState[ResponseT]) -> AgentRunMetrics | None:
        """Build aggregated run metrics from state.

        Args:
            state: The agent state with collected metrics.

        Returns:
            AgentRunMetrics instance, or None if state not initialized.
        """
        run_id = state.get(_KEY_RUN_ID)
        run_start = state.get(_KEY_RUN_START)

        if not run_id or run_start is None:
            return None

        now = time.perf_counter()
        elapsed = now - run_start
        start_datetime = datetime.now(tz=timezone.utc).replace(
            microsecond=int((datetime.now(tz=timezone.utc).timestamp() - elapsed) * 1_000_000)
            % 1_000_000
        )

        return AgentRunMetrics(
            run_id=run_id,
            start_time=start_datetime,
            end_time=datetime.now(tz=timezone.utc),
            model_calls=list(state.get(_KEY_MODEL_CALLS) or []),
            tool_calls=list(state.get(_KEY_TOOL_CALLS) or []),
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Track metrics for model calls (sync).

        Args:
            request: Model request containing model, messages, and state.
            handler: Callback to execute the model call.

        Returns:
            The model response from the handler.

        Raises:
            Exception: Re-raises any exception from the handler after recording metrics.
        """
        trace_ctx = self._build_trace_context(request)
        start_time = time.perf_counter()
        error: BaseException | None = None
        response: ModelResponse | None = None

        try:
            response = handler(request)
        except BaseException as e:
            error = e
            raise
        else:
            return response
        finally:
            metrics = self._build_model_call_metrics(
                request, response, start_time, error, trace_ctx
            )

            if self.exporter:
                self.exporter.export_model_call(metrics)

            self._store_model_metrics(request.state, metrics)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Track metrics for model calls (async).

        Args:
            request: Model request containing model, messages, and state.
            handler: Async callback to execute the model call.

        Returns:
            The model response from the handler.

        Raises:
            Exception: Re-raises any exception from the handler after recording metrics.
        """
        trace_ctx = self._build_trace_context(request)
        start_time = time.perf_counter()
        error: BaseException | None = None
        response: ModelResponse | None = None

        try:
            response = await handler(request)
        except BaseException as e:
            error = e
            raise
        else:
            return response
        finally:
            metrics = self._build_model_call_metrics(
                request, response, start_time, error, trace_ctx
            )

            if self.exporter:
                await self.exporter.aexport_model_call(metrics)

            self._store_model_metrics(request.state, metrics)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Track metrics for tool calls (sync).

        Args:
            request: Tool call request with tool info and state.
            handler: Callback to execute the tool call.

        Returns:
            The tool response from the handler.

        Raises:
            Exception: Re-raises any exception from the handler after recording metrics.
        """
        trace_ctx = self._build_trace_context(request)
        tool_name = request.tool.name if request.tool else request.tool_call.get("name", "unknown")
        start_time = time.perf_counter()
        error: BaseException | None = None

        try:
            response = handler(request)
        except BaseException as e:
            error = e
            raise
        else:
            return response
        finally:
            metrics = self._build_tool_call_metrics(tool_name, start_time, error, trace_ctx)

            if self.exporter:
                self.exporter.export_tool_call(metrics)

            self._store_tool_metrics(request.state, metrics)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Track metrics for tool calls (async).

        Args:
            request: Tool call request with tool info and state.
            handler: Async callback to execute the tool call.

        Returns:
            The tool response from the handler.

        Raises:
            Exception: Re-raises any exception from the handler after recording metrics.
        """
        trace_ctx = self._build_trace_context(request)
        tool_name = request.tool.name if request.tool else request.tool_call.get("name", "unknown")
        start_time = time.perf_counter()
        error: BaseException | None = None

        try:
            response = await handler(request)
        except BaseException as e:
            error = e
            raise
        else:
            return response
        finally:
            metrics = self._build_tool_call_metrics(tool_name, start_time, error, trace_ctx)

            if self.exporter:
                await self.exporter.aexport_tool_call(metrics)

            self._store_tool_metrics(request.state, metrics)

    @override
    def after_agent(
        self, state: MetricsState[ResponseT], runtime: Runtime
    ) -> dict[str, Any] | None:
        """Export aggregated run metrics when agent completes (sync).

        Args:
            state: The current agent state with collected metrics.
            runtime: The langgraph runtime.

        Returns:
            `None` (no state updates).
        """
        run_metrics = self._build_run_metrics(state)
        if run_metrics and self.exporter:
            self.exporter.export_run_complete(run_metrics)

        return None

    async def aafter_agent(
        self, state: MetricsState[ResponseT], runtime: Runtime
    ) -> dict[str, Any] | None:
        """Export aggregated run metrics when agent completes (async).

        Args:
            state: The current agent state with collected metrics.
            runtime: The langgraph runtime.

        Returns:
            `None` (no state updates).
        """
        run_metrics = self._build_run_metrics(state)
        if run_metrics and self.exporter:
            await self.exporter.aexport_run_complete(run_metrics)

        return None


__all__ = [
    "AgentRunMetrics",
    "CallbackMetricsExporter",
    "InMemoryMetricsExporter",
    "MetricsExporter",
    "MetricsMiddleware",
    "MetricsMultiExporter",
    "MetricsState",
    "ModelCallMetrics",
    "ToolCallMetrics",
]
