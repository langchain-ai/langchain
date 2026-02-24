"""Core types for the instrumentation layer.

Defines the Span and metric primitives that all instrumentation providers
must work with. Modeled after OpenTelemetry semantics but kept minimal
to avoid pulling in OTel as a hard dependency.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class SpanKind(enum.Enum):
    """The kind of work a span represents.

    Mirrors OpenTelemetry SpanKind semantics adapted for LLM pipelines.
    """

    CHAIN = "chain"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"
    AGENT = "agent"


class SpanStatus(enum.Enum):
    """Terminal status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanAttributes:
    """Structured attributes attached to a span.

    Provides typed fields for common LLM-pipeline attributes so providers
    don't need to parse opaque dicts.  Extra key-value pairs go into `extra`.
    """

    model_name: str | None = None
    provider: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    tool_name: str | None = None
    tool_call_count: int | None = None
    retriever_query: str | None = None
    retriever_top_k: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Span(Protocol):
    """Protocol that every instrumentation span must satisfy.

    Intentionally minimal — providers wrap their native span types
    (OTel Span, Datadog Span, etc.) behind this interface.
    """

    @property
    def span_id(self) -> str:
        """Unique identifier for this span."""
        ...

    @property
    def trace_id(self) -> str:
        """Identifier of the trace this span belongs to."""
        ...

    def set_attributes(self, attributes: SpanAttributes) -> None:
        """Attach structured attributes to the span.

        Args:
            attributes: The structured attributes to attach.
        """
        ...

    def set_status(self, status: SpanStatus, *, description: str = "") -> None:
        """Set the terminal status of the span.

        Args:
            status: The terminal status.
            description: Human-readable status description (typically the error message).
        """
        ...

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Record a timestamped event within the span.

        Useful for streaming tokens, retry attempts, etc.

        Args:
            name: Short event name (e.g. ``"new_token"``, ``"retry"``).
            attributes: Optional event-level key-value pairs.
        """
        ...

    def end(self) -> None:
        """Mark the span as finished.

        Must be called exactly once. Implementations should be idempotent
        if called more than once.
        """
        ...


@dataclass
class MetricEvent:
    """A single metric data point emitted by the pipeline.

    Providers receive these via `InstrumentationProvider.record_metric`
    and can forward them to Prometheus, StatsD, CloudWatch, etc.
    """

    name: str
    value: float
    unit: str = ""
    tags: dict[str, str] = field(default_factory=dict)


__all__ = [
    "MetricEvent",
    "Span",
    "SpanAttributes",
    "SpanKind",
    "SpanStatus",
]
