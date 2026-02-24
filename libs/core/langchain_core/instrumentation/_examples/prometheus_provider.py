"""Example: Prometheus metrics provider.

Translates `MetricEvent` data into Prometheus counters and histograms,
served on a configurable HTTP port via `prometheus_client`.

Requirements (not in core's dependencies):
    pip install prometheus-client

Usage::

    from langchain_core.instrumentation._examples.prometheus_provider import (
        PrometheusMetricsProvider,
    )
    provider = PrometheusMetricsProvider(port=9090)
    # → http://localhost:9090/metrics

See `full_setup.py` in this directory for a complete three-backend example.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.instrumentation.noop import NoopSpan
from langchain_core.instrumentation.types import (
    MetricEvent,
    SpanAttributes,
    SpanKind,
    SpanStatus,
)

logger = logging.getLogger(__name__)


class PrometheusMetricsProvider:
    """InstrumentationProvider that exposes Prometheus metrics.

    Automatically creates and serves counters/histograms from `MetricEvent`s.
    Span lifecycle is handled by returning `NoopSpan` — this provider only
    cares about metrics, not traces.
    """

    def __init__(self, port: int = 9090, *, prefix: str = "") -> None:
        try:
            from prometheus_client import (
                CollectorRegistry,
                Counter,
                Histogram,
                start_http_server,
            )
        except ImportError as exc:
            msg = (
                "prometheus-client is required for PrometheusMetricsProvider. "
                "Install it with: pip install prometheus-client"
            )
            raise ImportError(msg) from exc

        self._prefix = prefix
        self._registry = CollectorRegistry()
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()

        self._Counter = Counter
        self._Histogram = Histogram

        self._token_counter = Counter(
            f"{prefix}langchain_tokens_total",
            "Total LLM tokens consumed",
            ["model", "direction"],
            registry=self._registry,
        )
        self._llm_duration = Histogram(
            f"{prefix}langchain_llm_duration_seconds",
            "LLM call duration in seconds",
            ["model"],
            registry=self._registry,
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        )
        self._tool_duration = Histogram(
            f"{prefix}langchain_tool_duration_seconds",
            "Tool call duration in seconds",
            ["tool"],
            registry=self._registry,
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0),
        )
        self._chain_duration = Histogram(
            f"{prefix}langchain_chain_duration_seconds",
            "Chain call duration in seconds",
            ["chain"],
            registry=self._registry,
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
        )
        self._error_counter = Counter(
            f"{prefix}langchain_errors_total",
            "Total errors by component kind",
            ["kind"],
            registry=self._registry,
        )

        start_http_server(port, registry=self._registry)
        logger.info("Prometheus metrics server started on :%d", port)

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        *,
        parent: NoopSpan | None = None,
        attributes: SpanAttributes | None = None,
    ) -> _PrometheusTimingSpan:
        return _PrometheusTimingSpan(name, kind, self, attributes)

    def record_metric(self, event: MetricEvent) -> None:
        model = event.tags.get("model", "unknown")

        if "tokens.input" in event.name:
            self._token_counter.labels(model=model, direction="input").inc(event.value)
        elif "tokens.output" in event.name:
            self._token_counter.labels(model=model, direction="output").inc(event.value)
        elif "tokens.total" in event.name:
            self._token_counter.labels(model=model, direction="total").inc(event.value)

    def flush(self) -> None:
        pass

    def _observe_duration(
        self, kind: SpanKind, name: str, duration_s: float, error: bool,
    ) -> None:
        if kind in {SpanKind.LLM, SpanKind.EMBEDDING}:
            self._llm_duration.labels(model=name).observe(duration_s)
        elif kind == SpanKind.TOOL:
            self._tool_duration.labels(tool=name).observe(duration_s)
        elif kind in {SpanKind.CHAIN, SpanKind.AGENT}:
            self._chain_duration.labels(chain=name).observe(duration_s)

        if error:
            self._error_counter.labels(kind=kind.value).inc()


class _PrometheusTimingSpan:
    """Span that records its duration into Prometheus histograms on end."""

    __slots__ = (
        "_name", "_kind", "_provider", "_attributes",
        "_status", "_start_ns", "_ended",
    )

    def __init__(
        self,
        name: str,
        kind: SpanKind,
        provider: PrometheusMetricsProvider,
        attributes: SpanAttributes | None,
    ) -> None:
        import time
        self._name = name
        self._kind = kind
        self._provider = provider
        self._attributes = attributes or SpanAttributes()
        self._status = SpanStatus.UNSET
        self._start_ns = time.monotonic_ns()
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

        import time
        duration_s = (time.monotonic_ns() - self._start_ns) / 1_000_000_000

        label_name = self._attributes.model_name or self._attributes.tool_name or self._name
        self._provider._observe_duration(
            self._kind, label_name, duration_s,
            error=(self._status == SpanStatus.ERROR),
        )
