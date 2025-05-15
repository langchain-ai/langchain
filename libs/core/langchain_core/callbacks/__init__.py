"""**Callback handlers** allow listening to events in LangChain.

**Class hierarchy:**

.. code-block::

    BaseCallbackHandler --> <name>CallbackHandler  # Example: AimCallbackHandler
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.callbacks.base import (
        AsyncCallbackHandler,
        BaseCallbackHandler,
        BaseCallbackManager,
        CallbackManagerMixin,
        Callbacks,
        ChainManagerMixin,
        LLMManagerMixin,
        RetrieverManagerMixin,
        RunManagerMixin,
        ToolManagerMixin,
    )
    from langchain_core.callbacks.file import FileCallbackHandler
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
        ParentRunManager,
        RunManager,
        adispatch_custom_event,
        dispatch_custom_event,
    )
    from langchain_core.callbacks.stdout import StdOutCallbackHandler
    from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_core.callbacks.usage import (
        UsageMetadataCallbackHandler,
        get_usage_metadata_callback,
    )

__all__ = (
    "AsyncCallbackHandler",
    "AsyncCallbackManager",
    "AsyncCallbackManagerForChainGroup",
    "AsyncCallbackManagerForChainRun",
    "AsyncCallbackManagerForLLMRun",
    "AsyncCallbackManagerForRetrieverRun",
    "AsyncCallbackManagerForToolRun",
    "AsyncParentRunManager",
    "AsyncRunManager",
    "BaseCallbackHandler",
    "BaseCallbackManager",
    "BaseRunManager",
    "CallbackManager",
    "CallbackManagerForChainGroup",
    "CallbackManagerForChainRun",
    "CallbackManagerForLLMRun",
    "CallbackManagerForRetrieverRun",
    "CallbackManagerForToolRun",
    "CallbackManagerMixin",
    "Callbacks",
    "ChainManagerMixin",
    "FileCallbackHandler",
    "LLMManagerMixin",
    "ParentRunManager",
    "RetrieverManagerMixin",
    "RunManager",
    "RunManagerMixin",
    "StdOutCallbackHandler",
    "StreamingStdOutCallbackHandler",
    "ToolManagerMixin",
    "UsageMetadataCallbackHandler",
    "adispatch_custom_event",
    "dispatch_custom_event",
    "get_usage_metadata_callback",
)

_dynamic_imports = {
    "AsyncCallbackHandler": "base",
    "BaseCallbackHandler": "base",
    "BaseCallbackManager": "base",
    "CallbackManagerMixin": "base",
    "Callbacks": "base",
    "ChainManagerMixin": "base",
    "LLMManagerMixin": "base",
    "RetrieverManagerMixin": "base",
    "RunManagerMixin": "base",
    "ToolManagerMixin": "base",
    "FileCallbackHandler": "file",
    "AsyncCallbackManager": "manager",
    "AsyncCallbackManagerForChainGroup": "manager",
    "AsyncCallbackManagerForChainRun": "manager",
    "AsyncCallbackManagerForLLMRun": "manager",
    "AsyncCallbackManagerForRetrieverRun": "manager",
    "AsyncCallbackManagerForToolRun": "manager",
    "AsyncParentRunManager": "manager",
    "AsyncRunManager": "manager",
    "BaseRunManager": "manager",
    "CallbackManager": "manager",
    "CallbackManagerForChainGroup": "manager",
    "CallbackManagerForChainRun": "manager",
    "CallbackManagerForLLMRun": "manager",
    "CallbackManagerForRetrieverRun": "manager",
    "CallbackManagerForToolRun": "manager",
    "ParentRunManager": "manager",
    "RunManager": "manager",
    "adispatch_custom_event": "manager",
    "dispatch_custom_event": "manager",
    "StdOutCallbackHandler": "stdout",
    "StreamingStdOutCallbackHandler": "streaming_stdout",
    "UsageMetadataCallbackHandler": "usage",
    "get_usage_metadata_callback": "usage",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
