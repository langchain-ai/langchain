"""Tracers that record execution of LangChain runs."""

from typing import TYPE_CHECKING, Any

from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.langchain_v1 import LangChainTracerV1
from langchain_core.tracers.stdout import (
    ConsoleCallbackHandler,
    FunctionCallbackHandler,
)

from langchain._api import create_importer
from langchain.callbacks.tracers.logging import LoggingCallbackHandler

if TYPE_CHECKING:
    from langchain_community.callbacks.tracers.wandb import WandbTracer

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {"WandbTracer": "langchain_community.callbacks.tracers.wandb"}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "ConsoleCallbackHandler",
    "FunctionCallbackHandler",
    "LoggingCallbackHandler",
    "LangChainTracer",
    "LangChainTracerV1",
    "WandbTracer",
]
