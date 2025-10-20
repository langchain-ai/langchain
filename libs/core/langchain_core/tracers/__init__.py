"""**Tracers** are classes for tracing runs."""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.tracers.base import BaseTracer
    from langchain_core.tracers.evaluation import EvaluatorCallbackHandler
    from langchain_core.tracers.langchain import LangChainTracer
    from langchain_core.tracers.log_stream import (
        LogStreamCallbackHandler,
        RunLog,
        RunLogPatch,
    )
    from langchain_core.tracers.schemas import Run
    from langchain_core.tracers.stdout import ConsoleCallbackHandler

__all__ = (
    "BaseTracer",
    "ConsoleCallbackHandler",
    "EvaluatorCallbackHandler",
    "LangChainTracer",
    "LogStreamCallbackHandler",
    "Run",
    "RunLog",
    "RunLogPatch",
)

_dynamic_imports = {
    "BaseTracer": "base",
    "EvaluatorCallbackHandler": "evaluation",
    "LangChainTracer": "langchain",
    "LogStreamCallbackHandler": "log_stream",
    "RunLog": "log_stream",
    "RunLogPatch": "log_stream",
    "Run": "schemas",
    "ConsoleCallbackHandler": "stdout",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
