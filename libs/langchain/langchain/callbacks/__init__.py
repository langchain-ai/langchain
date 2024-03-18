"""**Callback handlers** allow listening to events in LangChain.

**Class hierarchy:**

.. code-block::

    BaseCallbackHandler --> <name>CallbackHandler  # Example: AimCallbackHandler
"""
from typing import Any

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

DEPRECATED_IMPORTS = [
    "AimCallbackHandler",
    "ArgillaCallbackHandler",
    "ArizeCallbackHandler",
    "PromptLayerCallbackHandler",
    "ArthurCallbackHandler",
    "ClearMLCallbackHandler",
    "CometCallbackHandler",
    "ContextCallbackHandler",
    "HumanApprovalCallbackHandler",
    "InfinoCallbackHandler",
    "MlflowCallbackHandler",
    "LLMonitorCallbackHandler",
    "OpenAICallbackHandler",
    "LLMThoughtLabeler",
    "wandb_tracing_enabled",
    "FlyteCallbackHandler",
    "SageMakerCallbackHandler",
    "LabelStudioCallbackHandler",
    "TrubricsCallbackHandler",
    "StreamlitCallbackHandler",
    "WandbCallbackHandler",
    "WhyLabsCallbackHandler",
    "get_openai_callback",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.callbacks import {name}`"
        )

    raise AttributeError()


__all__ = [
    "FileCallbackHandler",
    "StdOutCallbackHandler",
    "AsyncIteratorCallbackHandler",
    "StreamingStdOutCallbackHandler",
    "FinalStreamingStdOutCallbackHandler",
    "LangChainTracer",
    "tracing_enabled",
    "tracing_v2_enabled",
    "collect_runs",
]
