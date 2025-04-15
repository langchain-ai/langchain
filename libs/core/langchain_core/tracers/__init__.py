"""**Tracers** are classes for tracing runs.

**Class hierarchy:**

.. code-block::

    BaseCallbackHandler --> BaseTracer --> <name>Tracer  # Examples: LangChainTracer, RootListenersTracer
                                       --> <name>  # Examples: LogStreamCallbackHandler
"""  # noqa: E501

from importlib import import_module
from typing import TYPE_CHECKING

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

__all__ = [
    "BaseTracer",
    "EvaluatorCallbackHandler",
    "LangChainTracer",
    "ConsoleCallbackHandler",
    "Run",
    "RunLog",
    "RunLogPatch",
    "LogStreamCallbackHandler",
]

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
    package = __spec__.parent
    if module_name == "__module__" or module_name is None:
        result = import_module(f".{attr_name}", package=package)
    else:
        module = import_module(f".{module_name}", package=package)
        result = getattr(module, attr_name)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
