"""InstrumentationProvider protocol — the central abstraction.

Every observability backend (LangSmith, OpenTelemetry, Datadog, custom)
implements this protocol.  The core framework calls `start_span`,
`record_metric`, and `flush` — nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from langchain_core.instrumentation.types import (
        MetricEvent,
        Span,
        SpanAttributes,
        SpanKind,
    )


@runtime_checkable
class InstrumentationProvider(Protocol):
    """Protocol that observability backends must implement.

    Design goals:
    - Zero hard dependencies (no OTel, no LangSmith imports)
    - Supports both tracing (spans) and metrics in one interface
    - Providers are composable via `CompositeProvider`

    Example usage::

        provider = get_instrumentation_provider()
        span = provider.start_span("chat_model", SpanKind.LLM)
        span.set_attributes(SpanAttributes(model_name="gpt-4o"))
        try:
            result = call_model(...)
        except Exception as exc:
            span.set_status(SpanStatus.ERROR, description=str(exc))
            raise
        finally:
            span.end()
    """

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        *,
        parent: Span | None = None,
        attributes: SpanAttributes | None = None,
    ) -> Span:
        """Begin a new span.

        Args:
            name: Human-readable operation name (e.g. ``"ChatOpenAI"``).
            kind: The semantic kind of the span.
            parent: Optional parent span for nested traces.
            attributes: Initial attributes to attach.

        Returns:
            An active span that MUST be ended via `span.end()`.
        """
        ...

    def record_metric(self, event: MetricEvent) -> None:
        """Record a single metric data point.

        Called inline during execution. Implementations should be cheap
        (buffer + flush pattern recommended).

        Args:
            event: The metric event to record.
        """
        ...

    def flush(self) -> None:
        """Flush any buffered data to the backend.

        Called at the end of a top-level invocation to ensure all telemetry
        is delivered.  Implementations that send data synchronously on each
        call can make this a no-op.
        """
        ...


__all__ = [
    "InstrumentationProvider",
]
