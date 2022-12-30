"""Tracers that record execution of LangChain runs."""

from langchain.callbacks.tracers.base import Tracer, SharedTracer, JsonTracer, LangChainTracer


class SharedLangChainTracer(SharedTracer, LangChainTracer):
    """Shared tracer that records LangChain execution and POSTs to LangChain endpoint."""


class SharedJsonTracer(SharedTracer, JsonTracer):
    """Shared tracer that records LangChain execution as JSON."""
