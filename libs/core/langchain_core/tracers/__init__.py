"""**Tracers** are classes for tracing runs.

**Class hierarchy:**

.. code-block::

    BaseCallbackHandler --> BaseTracer --> <name>Tracer  # Examples: LangChainTracer, RootListenersTracer
                                       --> <name>  # Examples: LogStreamCallbackHandler
"""  # noqa: E501

from typing import TYPE_CHECKING

from langchain_core._lazy_imports import create_dynamic_getattr

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

__getattr__ = create_dynamic_getattr(
    package_name="langchain_core",
    module_path="tracers",
    dynamic_imports={
        "BaseTracer": "base",
        "EvaluatorCallbackHandler": "evaluation",
        "LangChainTracer": "langchain",
        "LogStreamCallbackHandler": "log_stream",
        "RunLog": "log_stream",
        "RunLogPatch": "log_stream",
        "Run": "schemas",
        "ConsoleCallbackHandler": "stdout",
    },
)


def __dir__() -> list[str]:
    return list(__all__)
