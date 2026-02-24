"""Composite provider that fans out to multiple backends.

Allows running LangSmith + OpenTelemetry + custom metrics simultaneously:

    set_instrumentation_provider(
        CompositeProvider([
            CallbackBridgeProvider(callback_manager),  # LangSmith via callbacks
            OTelProvider(tracer),                       # Jaeger / Tempo
            PrometheusMetricsProvider(),                # Prometheus counters
        ])
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.instrumentation.types import (
    MetricEvent,
    SpanAttributes,
    SpanKind,
    SpanStatus,
)

if TYPE_CHECKING:
    from langchain_core.instrumentation.provider import InstrumentationProvider
    from langchain_core.instrumentation.types import Span

logger = logging.getLogger(__name__)


class CompositeSpan:
    """A span that delegates to multiple underlying spans."""

    def __init__(self, spans: list[Span]) -> None:
        self._spans = spans

    @property
    def span_id(self) -> str:
        return self._spans[0].span_id if self._spans else ""

    @property
    def trace_id(self) -> str:
        return self._spans[0].trace_id if self._spans else ""

    def set_attributes(self, attributes: SpanAttributes) -> None:
        for span in self._spans:
            span.set_attributes(attributes)

    def set_status(self, status: SpanStatus, *, description: str = "") -> None:
        for span in self._spans:
            span.set_status(status, description=description)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        for span in self._spans:
            span.add_event(name, attributes=attributes)

    def end(self) -> None:
        for span in self._spans:
            try:
                span.end()
            except Exception:
                logger.warning(
                    "Error ending span %s in %s",
                    span.span_id,
                    type(span).__name__,
                    exc_info=True,
                )


# Need Any for the add_event attributes parameter
from typing import Any  # noqa: E402


class CompositeProvider:
    """Fans out instrumentation calls to multiple providers.

    Span lifecycle and metrics are forwarded to every registered provider.
    If one provider fails, the others still receive the call.
    """

    def __init__(self, providers: list[InstrumentationProvider]) -> None:
        self._providers = providers

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        *,
        parent: CompositeSpan | None = None,
        attributes: SpanAttributes | None = None,
    ) -> CompositeSpan:
        """Start a span on every registered provider.

        Args:
            name: Operation name.
            kind: Semantic span kind.
            parent: Optional composite parent span.
            attributes: Initial attributes.

        Returns:
            A composite span wrapping all provider spans.
        """
        spans: list[Span] = []
        for provider in self._providers:
            try:
                span = provider.start_span(
                    name, kind, parent=None, attributes=attributes,
                )
                spans.append(span)
            except Exception:
                logger.warning(
                    "Error starting span in %s",
                    type(provider).__name__,
                    exc_info=True,
                )
        return CompositeSpan(spans)

    def record_metric(self, event: MetricEvent) -> None:
        for provider in self._providers:
            try:
                provider.record_metric(event)
            except Exception:
                logger.warning(
                    "Error recording metric in %s",
                    type(provider).__name__,
                    exc_info=True,
                )

    def flush(self) -> None:
        for provider in self._providers:
            try:
                provider.flush()
            except Exception:
                logger.warning(
                    "Error flushing %s",
                    type(provider).__name__,
                    exc_info=True,
                )


__all__ = [
    "CompositeProvider",
    "CompositeSpan",
]
