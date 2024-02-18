"""**Callback handlers** allow listening to events in LangChain.

**Class hierarchy:**

.. code-block::

    BaseCallbackHandler --> <name>CallbackHandler  # Example: AimCallbackHandler
"""
import warnings
from typing import Any

from langchain_core._api import LangChainDeprecationWarning
from langchain_core.callbacks import (
    StdOutCallbackHandler,
    StreamingStdOutCallbackHandler,
)
from langchain_core.tracers.context import (
    collect_runs,
    tracing_enabled,
    tracing_v2_enabled,
)
from langchain_core.tracers.langchain import LangChainTracer

from langchain.callbacks.file import FileCallbackHandler
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.utils.interactive_env import is_interactive_env


def __getattr__(name: str) -> Any:
    from langchain_community import callbacks

    # If not in interactive env, raise warning.
    if not is_interactive_env():
        warnings.warn(
            "Importing this callback from langchain is deprecated. Importing it from "
            "langchain will no longer be supported as of langchain==0.2.0. "
            "Please import from langchain-community instead:\n\n"
            f"`from langchain_community.callbacks import {name}`.\n\n"
            "To install langchain-community run `pip install -U langchain-community`.",
            category=LangChainDeprecationWarning,
        )

    return getattr(callbacks, name)


__all__ = [
    "AimCallbackHandler",
    "ArgillaCallbackHandler",
    "ArizeCallbackHandler",
    "PromptLayerCallbackHandler",
    "ArthurCallbackHandler",
    "ClearMLCallbackHandler",
    "CometCallbackHandler",
    "ContextCallbackHandler",
    "FileCallbackHandler",
    "HumanApprovalCallbackHandler",
    "InfinoCallbackHandler",
    "MlflowCallbackHandler",
    "LLMonitorCallbackHandler",
    "OpenAICallbackHandler",
    "StdOutCallbackHandler",
    "AsyncIteratorCallbackHandler",
    "StreamingStdOutCallbackHandler",
    "FinalStreamingStdOutCallbackHandler",
    "LLMThoughtLabeler",
    "LangChainTracer",
    "StreamlitCallbackHandler",
    "WandbCallbackHandler",
    "WhyLabsCallbackHandler",
    "get_openai_callback",
    "tracing_enabled",
    "tracing_v2_enabled",
    "collect_runs",
    "wandb_tracing_enabled",
    "FlyteCallbackHandler",
    "SageMakerCallbackHandler",
    "LabelStudioCallbackHandler",
    "TrubricsCallbackHandler",
]
