"""Tracers that record execution of LangChain runs."""

from langchain.callbacks.tracers.base import SharedTracer, Tracer
from langchain.callbacks.tracers.langchain import BaseLangChainTracer


class SharedLangChainTracer(SharedTracer, BaseLangChainTracer):
    """Shared tracer that records LangChain execution to LangChain endpoint."""


class LangChainTracer(Tracer, BaseLangChainTracer):
    """Tracer that records LangChain execution to LangChain endpoint."""
