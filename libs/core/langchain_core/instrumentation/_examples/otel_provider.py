"""Example: OpenTelemetry instrumentation provider.

This module shows how a third-party or user can implement the
`InstrumentationProvider` protocol using OpenTelemetry.

NOT part of the public API — lives in `_examples/` to serve as
a reference implementation for the assignment.

Requirements (not added to core's dependencies):
    pip install opentelemetry-api opentelemetry-sdk

Usage::

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    # Setup OTel
    otel_provider = TracerProvider()
    otel_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(otel_provider)
    tracer = trace.get_tracer("langchain")

    # Plug into LangChain
    from langchain_core.instrumentation import set_instrumentation_provider
    from langchain_core.instrumentation._examples.otel_provider import OTelProvider

    set_instrumentation_provider(OTelProvider(tracer))

    # Now every LangChain operation emits OTel spans
    model.invoke("Hello")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.instrumentation.types import (
    MetricEvent,
    SpanAttributes,
    SpanKind,
    SpanStatus,
)

if TYPE_CHECKING:
    from opentelemetry import trace as otel_trace


_KIND_TO_OTEL_ATTR = {
    SpanKind.CHAIN: "langchain.chain",
    SpanKind.LLM: "langchain.llm",
    SpanKind.TOOL: "langchain.tool",
    SpanKind.RETRIEVER: "langchain.retriever",
    SpanKind.EMBEDDING: "langchain.embedding",
    SpanKind.AGENT: "langchain.agent",
}


class OTelSpan:
    """Wraps an OpenTelemetry span behind the LangChain `Span` protocol."""

    def __init__(self, otel_span: otel_trace.Span) -> None:
        self._span = otel_span
        ctx = otel_span.get_span_context()
        self._span_id = format(ctx.span_id, "016x") if ctx else ""
        self._trace_id = format(ctx.trace_id, "032x") if ctx else ""

    @property
    def span_id(self) -> str:
        return self._span_id

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def set_attributes(self, attributes: SpanAttributes) -> None:
        if attributes.model_name:
            self._span.set_attribute("gen_ai.request.model", attributes.model_name)
        if attributes.provider:
            self._span.set_attribute("gen_ai.system", attributes.provider)
        if attributes.temperature is not None:
            self._span.set_attribute(
                "gen_ai.request.temperature", attributes.temperature,
            )
        if attributes.max_tokens is not None:
            self._span.set_attribute(
                "gen_ai.request.max_tokens", attributes.max_tokens,
            )
        if attributes.input_tokens is not None:
            self._span.set_attribute(
                "gen_ai.usage.input_tokens", attributes.input_tokens,
            )
        if attributes.output_tokens is not None:
            self._span.set_attribute(
                "gen_ai.usage.output_tokens", attributes.output_tokens,
            )
        if attributes.total_tokens is not None:
            self._span.set_attribute(
                "gen_ai.usage.total_tokens", attributes.total_tokens,
            )
        if attributes.tool_name:
            self._span.set_attribute("langchain.tool.name", attributes.tool_name)
        if attributes.tool_call_count is not None:
            self._span.set_attribute(
                "langchain.tool_call_count", attributes.tool_call_count,
            )
        for key, value in attributes.extra.items():
            self._span.set_attribute(f"langchain.extra.{key}", value)

    def set_status(self, status: SpanStatus, *, description: str = "") -> None:
        from opentelemetry.trace import StatusCode

        mapping = {
            SpanStatus.UNSET: StatusCode.UNSET,
            SpanStatus.OK: StatusCode.OK,
            SpanStatus.ERROR: StatusCode.ERROR,
        }
        self._span.set_status(mapping[status], description=description)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._span.add_event(name, attributes=attributes or {})

    def end(self) -> None:
        self._span.end()


class OTelProvider:
    """OpenTelemetry implementation of `InstrumentationProvider`.

    Each `start_span` creates a real OTel span that flows through the
    configured OTel pipeline (processors → exporters → Jaeger/Tempo/etc.).
    """

    def __init__(self, tracer: otel_trace.Tracer) -> None:
        self._tracer = tracer

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        *,
        parent: OTelSpan | None = None,
        attributes: SpanAttributes | None = None,
    ) -> OTelSpan:
        from opentelemetry import context as otel_context
        from opentelemetry import trace as _trace

        ctx = None
        if parent is not None and isinstance(parent, OTelSpan):
            ctx = _trace.set_span_in_context(parent._span)

        otel_span = self._tracer.start_span(
            name,
            attributes={
                "langchain.span_kind": _KIND_TO_OTEL_ATTR.get(kind, "langchain.unknown"),
            },
            context=ctx,
        )
        span = OTelSpan(otel_span)
        if attributes:
            span.set_attributes(attributes)
        return span

    def record_metric(self, event: MetricEvent) -> None:
        pass

    def flush(self) -> None:
        pass
