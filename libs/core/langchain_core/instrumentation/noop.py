"""No-op implementations for zero-overhead defaults.

When no provider is configured, the framework uses `NoopProvider`.
All operations are no-ops with minimal overhead (no allocations, no I/O).
This guarantees that instrumentation code paths never add latency when
observability is disabled.
"""

from __future__ import annotations

from typing import Any

from langchain_core.instrumentation.types import (
    MetricEvent,
    SpanAttributes,
    SpanKind,
    SpanStatus,
)


class NoopSpan:
    """A span that does nothing.

    Returned by `NoopProvider.start_span`. All methods are no-ops.
    Satisfies the `Span` protocol without any side effects.
    """

    __slots__ = ()

    @property
    def span_id(self) -> str:
        return ""

    @property
    def trace_id(self) -> str:
        return ""

    def set_attributes(self, attributes: SpanAttributes) -> None:
        pass

    def set_status(self, status: SpanStatus, *, description: str = "") -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def end(self) -> None:
        pass


_NOOP_SPAN = NoopSpan()


class NoopProvider:
    """An instrumentation provider that discards everything.

    Used as the default when no observability backend is configured.
    Satisfies the `InstrumentationProvider` protocol.
    """

    __slots__ = ()

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        *,
        parent: NoopSpan | None = None,
        attributes: SpanAttributes | None = None,
    ) -> NoopSpan:
        return _NOOP_SPAN

    def record_metric(self, event: MetricEvent) -> None:
        pass

    def flush(self) -> None:
        pass


__all__ = [
    "NoopProvider",
    "NoopSpan",
]
