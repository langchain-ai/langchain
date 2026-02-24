"""Bridge between the new InstrumentationProvider and the legacy callback system.

`CallbackBridgeProvider` wraps an existing `CallbackManager` so that code
written against the new `InstrumentationProvider` protocol still fires
the classic callback events.  This is the backward-compatibility layer
that lets us migrate incrementally:

    Old code (callbacks) → works unchanged
    New code (provider)  → CallbackBridgeProvider → callbacks → handlers

When users switch to a native provider (e.g. OTel), the bridge is no longer
needed and callbacks become optional.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain_core.instrumentation.types import (
    MetricEvent,
    SpanAttributes,
    SpanKind,
    SpanStatus,
)

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        CallbackManager,
        CallbackManagerForChainRun,
        CallbackManagerForLLMRun,
        CallbackManagerForToolRun,
    )

logger = logging.getLogger(__name__)

_KIND_TO_RUN_TYPE: dict[SpanKind, str] = {
    SpanKind.CHAIN: "chain",
    SpanKind.LLM: "llm",
    SpanKind.TOOL: "tool",
    SpanKind.RETRIEVER: "retriever",
    SpanKind.EMBEDDING: "llm",
    SpanKind.AGENT: "chain",
}


class CallbackBridgeSpan:
    """A span backed by a CallbackManager run.

    Translates `Span` protocol calls into the corresponding callback events
    (`on_chain_start` / `on_chain_end`, `on_llm_start` / `on_llm_end`, etc.).
    """

    def __init__(
        self,
        *,
        name: str,
        kind: SpanKind,
        run_manager: (
            CallbackManagerForChainRun
            | CallbackManagerForLLMRun
            | CallbackManagerForToolRun
        ),
        run_id: str,
        trace_id: str,
    ) -> None:
        self._name = name
        self._kind = kind
        self._run_manager = run_manager
        self._run_id = run_id
        self._trace_id = trace_id
        self._ended = False
        self._attributes = SpanAttributes()
        self._status = SpanStatus.UNSET

    @property
    def span_id(self) -> str:
        return self._run_id

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def set_attributes(self, attributes: SpanAttributes) -> None:
        self._attributes = attributes

    def set_status(self, status: SpanStatus, *, description: str = "") -> None:
        self._status = status
        if status == SpanStatus.ERROR and description:
            self._attributes.error_message = description

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True

        if self._status == SpanStatus.ERROR:
            error_msg = self._attributes.error_message or "unknown error"
            _dispatch_error(self._run_manager, self._kind, error_msg)
        else:
            _dispatch_end(self._run_manager, self._kind)


def _dispatch_end(
    run_manager: Any,
    kind: SpanKind,
) -> None:
    """Dispatch the appropriate on_*_end callback."""
    if kind in {SpanKind.CHAIN, SpanKind.AGENT}:
        run_manager.on_chain_end(outputs={})
    elif kind in {SpanKind.LLM, SpanKind.EMBEDDING}:
        pass
    elif kind == SpanKind.TOOL:
        run_manager.on_tool_end(output="")
    elif kind == SpanKind.RETRIEVER:
        run_manager.on_retriever_end(documents=[])


def _dispatch_error(
    run_manager: Any,
    kind: SpanKind,
    error_message: str,
) -> None:
    """Dispatch the appropriate on_*_error callback."""
    error = RuntimeError(error_message)
    if kind in {SpanKind.CHAIN, SpanKind.AGENT}:
        run_manager.on_chain_error(error)
    elif kind in {SpanKind.LLM, SpanKind.EMBEDDING}:
        run_manager.on_llm_error(error)
    elif kind == SpanKind.TOOL:
        run_manager.on_tool_error(error)
    elif kind == SpanKind.RETRIEVER:
        run_manager.on_retriever_error(error)


class CallbackBridgeProvider:
    """Adapts legacy `CallbackManager` to the `InstrumentationProvider` protocol.

    Use this when migrating existing code: the provider interface is new,
    but the actual telemetry still flows through callbacks to existing handlers
    (LangChainTracer, ConsoleCallbackHandler, etc.).

    Example::

        from langchain_core.callbacks import CallbackManager
        from langchain_core.instrumentation import CallbackBridgeProvider

        cb_manager = CallbackManager(handlers=[my_handler])
        provider = CallbackBridgeProvider(cb_manager)

        span = provider.start_span("my_chain", SpanKind.CHAIN)
        # ... do work ...
        span.end()  # fires on_chain_end on my_handler
    """

    def __init__(self, callback_manager: CallbackManager) -> None:
        self._callback_manager = callback_manager

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        *,
        parent: CallbackBridgeSpan | None = None,
        attributes: SpanAttributes | None = None,
    ) -> CallbackBridgeSpan:
        """Start a new span by firing the appropriate on_*_start callback.

        Args:
            name: Operation name.
            kind: The semantic kind of span.
            parent: Optional parent span (unused in bridge; parent-child is
                managed by the callback manager's `parent_run_id`).
            attributes: Initial span attributes.

        Returns:
            A bridge span wrapping the callback run manager.
        """
        run_id = uuid.uuid4()
        trace_id = parent.trace_id if parent else str(run_id)

        serialized = {"name": name, "id": [str(run_id)]}
        run_manager = _dispatch_start(
            self._callback_manager, kind, name, serialized, run_id,
        )
        span = CallbackBridgeSpan(
            name=name,
            kind=kind,
            run_manager=run_manager,
            run_id=str(run_id),
            trace_id=trace_id,
        )
        if attributes:
            span.set_attributes(attributes)
        return span

    def record_metric(self, event: MetricEvent) -> None:
        pass

    def flush(self) -> None:
        pass


def _dispatch_start(
    callback_manager: CallbackManager,
    kind: SpanKind,
    name: str,
    serialized: dict[str, Any],
    run_id: uuid.UUID,
) -> Any:
    """Dispatch the appropriate on_*_start callback and return the run manager."""
    if kind in {SpanKind.CHAIN, SpanKind.AGENT}:
        return callback_manager.on_chain_start(
            serialized, inputs={}, run_id=run_id, name=name,
        )
    elif kind in {SpanKind.LLM, SpanKind.EMBEDDING}:
        return callback_manager.on_llm_start(
            serialized, prompts=[], run_id=run_id, name=name,
        )
    elif kind == SpanKind.TOOL:
        return callback_manager.on_tool_start(
            serialized, input_str="", run_id=run_id, name=name,
        )
    elif kind == SpanKind.RETRIEVER:
        return callback_manager.on_retriever_start(
            serialized, query="", run_id=run_id, name=name,
        )
    msg = f"Unsupported span kind: {kind}"
    raise ValueError(msg)


__all__ = [
    "CallbackBridgeProvider",
    "CallbackBridgeSpan",
]
