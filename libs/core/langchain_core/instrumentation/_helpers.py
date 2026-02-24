"""Internal helpers for integrating instrumentation into framework internals.

These functions are NOT part of the public API. They're used by
`BaseChatModel`, `BaseTool`, and `Runnable._call_with_config` to emit
spans and metrics without duplicating boilerplate.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

from langchain_core.instrumentation.config import get_instrumentation_provider
from langchain_core.instrumentation.types import (
    MetricEvent,
    SpanAttributes,
    SpanKind,
    SpanStatus,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from langchain_core.instrumentation.types import Span

logger = logging.getLogger(__name__)

_current_span: ContextVar[Span | None] = ContextVar("_current_span", default=None)


def get_current_span() -> Span | None:
    """Return the currently active instrumentation span, if any.

    Returns:
        The active span, or `None` if no span is in progress.
    """
    return _current_span.get()


@contextmanager
def instrumented_span(
    name: str,
    kind: SpanKind,
    *,
    parent: Span | None = None,
    attributes: SpanAttributes | None = None,
) -> Generator[Span, None, None]:
    """Context manager that wraps an operation in an instrumentation span.

    Automatically inherits the parent span from context if not provided
    explicitly.  Sets status to OK on normal exit, ERROR on exception,
    and always calls `span.end()`.

    Args:
        name: Human-readable operation name.
        kind: Semantic kind of the span.
        parent: Explicit parent span. If `None`, the current context span
            is used automatically.
        attributes: Initial attributes to attach.

    Yields:
        The active span — callers can add attributes/events during execution.
    """
    if parent is None:
        parent = _current_span.get()

    provider = get_instrumentation_provider()
    span = provider.start_span(name, kind, parent=parent, attributes=attributes)
    token: Token[Span | None] = _current_span.set(span)
    try:
        yield span
    except BaseException as e:
        span.set_status(SpanStatus.ERROR, description=str(e))
        raise
    else:
        span.set_status(SpanStatus.OK)
    finally:
        span.end()
        _current_span.reset(token)


def extract_llm_attributes(
    model: Any,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> SpanAttributes:
    """Extract LLM span attributes from a chat model instance.

    Pulls model_name, provider, temperature, and max_tokens from the model
    object or kwargs, mirroring the logic in `_get_ls_params` but producing
    a vendor-neutral `SpanAttributes`.

    Args:
        model: The language model instance (BaseChatModel / BaseLLM).
        stop: Stop sequences, if any.
        **kwargs: Additional invocation kwargs that may override model attributes.

    Returns:
        Populated span attributes.
    """
    model_name = _resolve_attr(model, kwargs, "model", "model_name")
    provider_name = _resolve_provider_name(model)
    temperature = _resolve_numeric_attr(model, kwargs, "temperature")
    max_tokens = _resolve_int_attr(model, kwargs, "max_tokens")

    return SpanAttributes(
        model_name=model_name,
        provider=provider_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def record_token_usage(
    span: Span,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    model_name: str | None = None,
) -> None:
    """Update span attributes with token usage and emit metric events.

    Args:
        span: The active span to update.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        total_tokens: Total token count.
        model_name: Model name for metric tags.
    """
    attrs = SpanAttributes(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    span.set_attributes(attrs)

    provider = get_instrumentation_provider()
    tags = {"model": model_name} if model_name else {}

    if input_tokens is not None:
        provider.record_metric(MetricEvent(
            name="langchain.llm.tokens.input",
            value=float(input_tokens),
            unit="tokens",
            tags=tags,
        ))
    if output_tokens is not None:
        provider.record_metric(MetricEvent(
            name="langchain.llm.tokens.output",
            value=float(output_tokens),
            unit="tokens",
            tags=tags,
        ))
    if total_tokens is not None:
        provider.record_metric(MetricEvent(
            name="langchain.llm.tokens.total",
            value=float(total_tokens),
            unit="tokens",
            tags=tags,
        ))


def extract_tool_attributes(tool: Any) -> SpanAttributes:
    """Extract span attributes from a tool instance.

    Args:
        tool: The BaseTool instance.

    Returns:
        Populated span attributes with tool name.
    """
    return SpanAttributes(
        tool_name=getattr(tool, "name", type(tool).__name__),
    )


# ── private helpers ──


def _resolve_attr(obj: Any, kwargs: dict[str, Any], *names: str) -> str | None:
    for name in names:
        if name in kwargs and isinstance(kwargs[name], str):
            return kwargs[name]
    for name in names:
        val = getattr(obj, name, None)
        if isinstance(val, str):
            return val
    return None


def _resolve_provider_name(model: Any) -> str | None:
    cls_name = type(model).__name__
    if cls_name.startswith("Chat"):
        return cls_name[4:].lower()
    if cls_name.endswith("Chat"):
        return cls_name[:-4].lower()
    return cls_name.lower()


def _resolve_numeric_attr(
    obj: Any, kwargs: dict[str, Any], name: str,
) -> float | None:
    if name in kwargs and isinstance(kwargs[name], (int, float)):
        return float(kwargs[name])
    val = getattr(obj, name, None)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _resolve_int_attr(
    obj: Any, kwargs: dict[str, Any], name: str,
) -> int | None:
    if name in kwargs and isinstance(kwargs[name], int):
        return kwargs[name]
    val = getattr(obj, name, None)
    if isinstance(val, int):
        return val
    return None
