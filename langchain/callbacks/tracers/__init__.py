"""Tracers that record execution of LangChain runs."""

from langchain.callbacks.tracers.base import (
    BaseJsonTracer,
    BaseLangChainTracer,
    SharedTracer,
    Tracer,
)


class SharedLangChainTracer(SharedTracer, BaseLangChainTracer):
    """Shared tracer that records LangChain execution and POSTs to LangChain endpoint."""


class SharedJsonTracer(SharedTracer, BaseJsonTracer):
    """Shared tracer that records LangChain execution as JSON."""


class LangChainTracer(Tracer, BaseLangChainTracer):
    """Tracer that records LangChain execution and POSTs to LangChain endpoint."""


class JsonTracer(Tracer, BaseJsonTracer):
    """Tracer that records LangChain execution as JSON."""
