from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainGroup,
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForLLMRun,
    AsyncCallbackManagerForRetrieverRun,
    AsyncCallbackManagerForToolRun,
    AsyncParentRunManager,
    AsyncRunManager,
    BaseRunManager,
    CallbackManager,
    CallbackManagerForChainGroup,
    CallbackManagerForChainRun,
    CallbackManagerForLLMRun,
    CallbackManagerForRetrieverRun,
    CallbackManagerForToolRun,
    Callbacks,
    ParentRunManager,
    RunManager,
    ahandle_event,
    atrace_as_chain_group,
    handle_event,
    trace_as_chain_group,
)
from langchain_core.tracers.context import (
    collect_runs,
    tracing_enabled,
    tracing_v2_enabled,
)
from langchain_core.utils.env import env_var_is_set

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.callbacks.manager import (
        get_openai_callback,
        wandb_tracing_enabled,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "get_openai_callback": "langchain_community.callbacks.manager",
    "wandb_tracing_enabled": "langchain_community.callbacks.manager",
}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "ahandle_event",
    "AsyncCallbackManagerForChainGroup",
    "AsyncCallbackManagerForChainRun",
    "AsyncCallbackManagerForLLMRun",
    "AsyncCallbackManagerForRetrieverRun",
    "AsyncCallbackManagerForToolRun",
    "AsyncParentRunManager",
    "AsyncRunManager",
    "atrace_as_chain_group",
    "BaseRunManager",
    "CallbackManager",
    "CallbackManagerForChainGroup",
    "CallbackManagerForChainRun",
    "CallbackManagerForLLMRun",
    "CallbackManagerForRetrieverRun",
    "CallbackManagerForToolRun",
    "Callbacks",
    "AsyncCallbackManager",
    "collect_runs",
    "env_var_is_set",
    "get_openai_callback",
    "handle_event",
    "ParentRunManager",
    "RunManager",
    "trace_as_chain_group",
    "tracing_enabled",
    "tracing_v2_enabled",
    "wandb_tracing_enabled",
]
