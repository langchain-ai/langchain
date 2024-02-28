"""LangChain **Runnable** and the **LangChain Expression Language (LCEL)**.

The LangChain Expression Language (LCEL) offers a declarative method to build
production-grade programs that harness the power of LLMs.

Programs created using LCEL and LangChain Runnables inherently support
synchronous, asynchronous, batch, and streaming operations.

Support for **async** allows servers hosting LCEL based programs to scale better
for higher concurrent loads.

**Streaming** of intermediate outputs as they're being generated allows for
creating more responsive UX.

This module contains schema and implementation of LangChain Runnables primitives.
"""
from langchain_core.runnables.base import (
    Runnable,
    RunnableBinding,
    RunnableGenerator,
    RunnableLambda,
    RunnableMap,
    RunnableParallel,
    RunnableSequence,
    RunnableSerializable,
)
from langchain_core.runnables.branch import RunnableBranch
from langchain_core.runnables.config import RunnableConfig, patch_config
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.router import RouterInput, RouterRunnable
from langchain_core.runnables.utils import (
    ConfigurableField,
    ConfigurableFieldMultiOption,
    ConfigurableFieldSingleOption,
)

__all__ = [
    "ConfigurableField",
    "ConfigurableFieldSingleOption",
    "ConfigurableFieldMultiOption",
    "patch_config",
    "RouterInput",
    "RouterRunnable",
    "Runnable",
    "RunnableSerializable",
    "RunnableBinding",
    "RunnableBranch",
    "RunnableConfig",
    "RunnableGenerator",
    "RunnableLambda",
    "RunnableMap",
    "RunnableParallel",
    "RunnablePassthrough",
    "RunnableSequence",
    "RunnableWithFallbacks",
]
