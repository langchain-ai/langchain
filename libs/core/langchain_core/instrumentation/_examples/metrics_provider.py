"""Example: Standalone metrics provider (no tracing).

Shows how the `InstrumentationProvider` protocol separates metrics from
tracing.  A team that only needs Prometheus counters doesn't need to
configure tracing at all.

Usage::

    from langchain_core.instrumentation import set_instrumentation_provider
    from langchain_core.instrumentation._examples.metrics_provider import (
        InMemoryMetricsProvider,
    )

    metrics = InMemoryMetricsProvider()
    set_instrumentation_provider(metrics)

    model.invoke("Hello")

    # After invocation, inspect collected metrics:
    for m in metrics.get_metrics():
        print(f"{m.name} = {m.value} {m.unit}  tags={m.tags}")

    # Example output:
    #   langchain.llm.tokens.input = 5.0   tags={'model': 'gpt-4o'}
    #   langchain.llm.tokens.output = 12.0  tags={'model': 'gpt-4o'}
    #   langchain.llm.duration_ms = 340.0   tags={'model': 'gpt-4o'}
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.instrumentation.noop import NoopSpan
from langchain_core.instrumentation.types import (
    MetricEvent,
    SpanAttributes,
    SpanKind,
    SpanStatus,
)


class InMemoryMetricsProvider:
    """Collects metrics in memory for inspection or forwarding.

    This is a minimal example showing how metrics flow through the provider
    interface independently of spans/tracing.  In production, you'd replace
    the in-memory buffer with a Prometheus client, StatsD, or CloudWatch.
    """

    def __init__(self) -> None:
        self._metrics: list[MetricEvent] = []
        self._lock = threading.Lock()

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        *,
        parent: NoopSpan | None = None,
        attributes: SpanAttributes | None = None,
    ) -> NoopSpan:
        return NoopSpan()

    def record_metric(self, event: MetricEvent) -> None:
        with self._lock:
            self._metrics.append(event)

    def flush(self) -> None:
        pass

    def get_metrics(self) -> list[MetricEvent]:
        """Return a snapshot of all collected metrics."""
        with self._lock:
            return list(self._metrics)

    def clear(self) -> None:
        """Clear collected metrics."""
        with self._lock:
            self._metrics.clear()


class TimingSpan:
    """A span that only records duration as a metric on end.

    Demonstrates that spans and metrics can work together: the span
    records its duration as a `MetricEvent` when `end()` is called.
    """

    def __init__(
        self,
        name: str,
        kind: SpanKind,
        provider: TimingMetricsProvider,
    ) -> None:
        self._name = name
        self._kind = kind
        self._provider = provider
        self._attributes = SpanAttributes()
        self._status = SpanStatus.UNSET
        self._start_ns = _monotonic_ns()
        self._ended = False

    @property
    def span_id(self) -> str:
        return ""

    @property
    def trace_id(self) -> str:
        return ""

    def set_attributes(self, attributes: SpanAttributes) -> None:
        self._attributes = attributes

    def set_status(self, status: SpanStatus, *, description: str = "") -> None:
        self._status = status

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True

        duration_ms = (_monotonic_ns() - self._start_ns) / 1_000_000

        tags: dict[str, str] = {"kind": self._kind.value, "status": self._status.value}
        if self._attributes.model_name:
            tags["model"] = self._attributes.model_name
        if self._attributes.tool_name:
            tags["tool"] = self._attributes.tool_name

        self._provider.record_metric(MetricEvent(
            name=f"langchain.{self._kind.value}.duration_ms",
            value=duration_ms,
            unit="ms",
            tags=tags,
        ))

        if self._attributes.input_tokens is not None:
            self._provider.record_metric(MetricEvent(
                name=f"langchain.{self._kind.value}.tokens.input",
                value=float(self._attributes.input_tokens),
                unit="tokens",
                tags=tags,
            ))
        if self._attributes.output_tokens is not None:
            self._provider.record_metric(MetricEvent(
                name=f"langchain.{self._kind.value}.tokens.output",
                value=float(self._attributes.output_tokens),
                unit="tokens",
                tags=tags,
            ))


class TimingMetricsProvider(InMemoryMetricsProvider):
    """Extends `InMemoryMetricsProvider` with auto-duration spans.

    Every span automatically emits a `duration_ms` metric and optional
    token count metrics when ended.
    """

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        *,
        parent: TimingSpan | None = None,
        attributes: SpanAttributes | None = None,
    ) -> TimingSpan:
        span = TimingSpan(name, kind, self)
        if attributes:
            span.set_attributes(attributes)
        return span


def _monotonic_ns() -> int:
    """Return monotonic clock in nanoseconds."""
    import time
    return time.monotonic_ns()
