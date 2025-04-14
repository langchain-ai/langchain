"""LangChain **Runnable** and the **LangChain Expression Language (LCEL)**.

The LangChain Expression Language (LCEL) offers a declarative method to build
production-grade programs that harness the power of LLMs.

Programs created using LCEL and LangChain Runnables inherently support
synchronous, asynchronous, batch, and streaming operations.

Support for **async** allows servers hosting LCEL based programs to scale better
for higher concurrent loads.

**Batch** operations allow for processing multiple inputs in parallel.

**Streaming** of intermediate outputs, as they're being generated, allows for
creating more responsive UX.

This module contains schema and implementation of LangChain Runnables primitives.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables.base import (
        Runnable,
        RunnableBinding,
        RunnableGenerator,
        RunnableLambda,
        RunnableMap,
        RunnableParallel,
        RunnableSequence,
        RunnableSerializable,
        chain,
    )
    from langchain_core.runnables.branch import RunnableBranch
    from langchain_core.runnables.config import (
        RunnableConfig,
        ensure_config,
        get_config_list,
        patch_config,
        run_in_executor,
    )
    from langchain_core.runnables.fallbacks import RunnableWithFallbacks
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.runnables.passthrough import (
        RunnableAssign,
        RunnablePassthrough,
        RunnablePick,
    )
    from langchain_core.runnables.router import RouterInput, RouterRunnable
    from langchain_core.runnables.utils import (
        AddableDict,
        ConfigurableField,
        ConfigurableFieldMultiOption,
        ConfigurableFieldSingleOption,
        ConfigurableFieldSpec,
        aadd,
        add,
    )

__all__ = [
    "chain",
    "AddableDict",
    "ConfigurableField",
    "ConfigurableFieldSingleOption",
    "ConfigurableFieldMultiOption",
    "ConfigurableFieldSpec",
    "ensure_config",
    "run_in_executor",
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
    "RunnableAssign",
    "RunnablePick",
    "RunnableSequence",
    "RunnableWithFallbacks",
    "RunnableWithMessageHistory",
    "get_config_list",
    "aadd",
    "add",
]

_dynamic_imports = {
    "chain": "base",
    "Runnable": "base",
    "RunnableBinding": "base",
    "RunnableGenerator": "base",
    "RunnableLambda": "base",
    "RunnableMap": "base",
    "RunnableParallel": "base",
    "RunnableSequence": "base",
    "RunnableSerializable": "base",
    "RunnableBranch": "branch",
    "RunnableConfig": "config",
    "ensure_config": "config",
    "get_config_list": "config",
    "patch_config": "config",
    "run_in_executor": "config",
    "RunnableWithFallbacks": "fallbacks",
    "RunnableWithMessageHistory": "history",
    "RunnableAssign": "passthrough",
    "RunnablePassthrough": "passthrough",
    "RunnablePick": "passthrough",
    "RouterInput": "router",
    "RouterRunnable": "router",
    "AddableDict": "utils",
    "ConfigurableField": "utils",
    "ConfigurableFieldMultiOption": "utils",
    "ConfigurableFieldSingleOption": "utils",
    "ConfigurableFieldSpec": "utils",
    "aadd": "utils",
    "add": "utils",
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
