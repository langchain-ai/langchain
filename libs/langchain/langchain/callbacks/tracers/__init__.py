"""Tracers that record execution of LangChain runs."""
from typing import Any

from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.langchain_v1 import LangChainTracerV1
from langchain_core.tracers.stdout import (
    ConsoleCallbackHandler,
    FunctionCallbackHandler,
)

from langchain.callbacks.tracers.logging import LoggingCallbackHandler

DEPRECATED_IMPORTS = [
    "WandbTracer",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.callbacks.tracers import {name}`"
        )

    raise AttributeError()


__all__ = [
    "ConsoleCallbackHandler",
    "FunctionCallbackHandler",
    "LoggingCallbackHandler",
    "LangChainTracer",
    "LangChainTracerV1",
]
