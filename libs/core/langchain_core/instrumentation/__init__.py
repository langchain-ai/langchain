"""Pluggable instrumentation layer for LangChain.

This package provides a provider-agnostic interface for tracing and metrics
collection.  Out of the box, `NoopProvider` is active (zero overhead).
Users can plug in any backend by calling `set_instrumentation_provider`.

Quick start::

    from langchain_core.instrumentation import (
        set_instrumentation_provider,
        get_instrumentation_provider,
        SpanKind,
        SpanAttributes,
        SpanStatus,
    )

    # At application startup
    set_instrumentation_provider(my_otel_provider)

    # In framework internals or user code
    provider = get_instrumentation_provider()
    span = provider.start_span("my_operation", SpanKind.CHAIN)
    try:
        result = do_work()
        span.set_status(SpanStatus.OK)
    except Exception as exc:
        span.set_status(SpanStatus.ERROR, description=str(exc))
        raise
    finally:
        span.end()

Backward compatibility:
    The existing callback system continues to work unchanged.
    `CallbackBridgeProvider` wraps a `CallbackManager` so new provider-based
    code still fires legacy callback events.
"""

from langchain_core.instrumentation.callback_bridge import (
    CallbackBridgeProvider,
    CallbackBridgeSpan,
)
from langchain_core.instrumentation.composite import (
    CompositeProvider,
    CompositeSpan,
)
from langchain_core.instrumentation.config import (
    get_instrumentation_provider,
    reset_instrumentation_provider,
    set_instrumentation_provider,
)
from langchain_core.instrumentation.noop import NoopProvider, NoopSpan
from langchain_core.instrumentation.provider import InstrumentationProvider
from langchain_core.instrumentation.types import (
    MetricEvent,
    Span,
    SpanAttributes,
    SpanKind,
    SpanStatus,
)

__all__ = [
    "CallbackBridgeProvider",
    "CallbackBridgeSpan",
    "CompositeProvider",
    "CompositeSpan",
    "InstrumentationProvider",
    "MetricEvent",
    "NoopProvider",
    "NoopSpan",
    "Span",
    "SpanAttributes",
    "SpanKind",
    "SpanStatus",
    "get_instrumentation_provider",
    "reset_instrumentation_provider",
    "set_instrumentation_provider",
]
