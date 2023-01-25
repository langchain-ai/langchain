"""Tracers that record execution of LangChain runs."""

from langchain.callbacks.tracers.base import BaseLangChainTracer, SharedTracer, Tracer


class SharedLangChainTracer(SharedTracer, BaseLangChainTracer):
    """Shared tracer that records LangChain execution and POSTs to LangChain endpoint."""


class LangChainTracer(Tracer, BaseLangChainTracer):
    """Tracer that records LangChain execution and POSTs to LangChain endpoint."""
