"""**Callback handlers** allow listening to events in LangChain.

**Class hierarchy:**

.. code-block::

    BaseCallbackHandler --> <name>CallbackHandler  # Example: AimCallbackHandler
"""
from typing import TYPE_CHECKING, Any

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

def __getattr__(name: str) -> Any:
    if name == "dispatch_custom_event":
        from langchain_core.callbacks.manager import dispatch_custom_event

        return dispatch_custom_event
    if name == "adispatch_custom_event":
        from langchain_core.callbacks.manager import adispatch_custom_event

        return adispatch_custom_event
    if name == "RetrieverManagerMixin":
        from langchain_core.callbacks.base import RetrieverManagerMixin

        return RetrieverManagerMixin
    if name == "LLMManagerMixin":
        from langchain_core.callbacks.base import LLMManagerMixin

        return LLMManagerMixin
    if name == "ChainManagerMixin":
        from langchain_core.callbacks.base import ChainManagerMixin

        return ChainManagerMixin
    if name == "ToolManagerMixin":
        from langchain_core.callbacks.base import ToolManagerMixin

        return ToolManagerMixin
    if name == "Callbacks":
        from langchain_core.callbacks.base import Callbacks

        return Callbacks
    if name == "CallbackManagerMixin":
        from langchain_core.callbacks.base import CallbackManagerMixin

        return CallbackManagerMixin
    if name == "RunManagerMixin":
        from langchain_core.callbacks.base import RunManagerMixin

        return RunManagerMixin
    if name == "BaseCallbackHandler":
        from langchain_core.callbacks.base import BaseCallbackHandler

        return BaseCallbackHandler
    if name == "AsyncCallbackHandler":
        from langchain_core.callbacks.base import AsyncCallbackHandler

        return AsyncCallbackHandler
    if name == "BaseCallbackManager":
        from langchain_core.callbacks.manager import BaseCallbackManager

        return BaseCallbackManager
    if name == "BaseRunManager":
        from langchain_core.callbacks.manager import BaseRunManager

        return BaseRunManager
    if name == "RunManager":
        from langchain_core.callbacks.manager import RunManager

        return RunManager
    if name == "ParentRunManager":
        from langchain_core.callbacks.manager import ParentRunManager

        return ParentRunManager
    if name == "AsyncRunManager":
        from langchain_core.callbacks.manager import AsyncRunManager

        return AsyncRunManager
    if name == "AsyncParentRunManager":
        from langchain_core.callbacks.manager import AsyncParentRunManager

        return AsyncParentRunManager
    if name == "CallbackManagerForLLMRun":
        from langchain_core.callbacks.manager import CallbackManagerForLLMRun

        return CallbackManagerForLLMRun
    if name == "AsyncCallbackManagerForLLMRun":
        from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun

        return AsyncCallbackManagerForLLMRun
    if name == "CallbackManagerForChainRun":
        from langchain_core.callbacks.manager import CallbackManagerForChainRun

        return CallbackManagerForChainRun
    if name == "AsyncCallbackManagerForChainRun":
        from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun

        return AsyncCallbackManagerForChainRun
    if name == "CallbackManagerForToolRun":
        from langchain_core.callbacks.manager import CallbackManagerForToolRun

        return CallbackManagerForToolRun
    if name == "AsyncCallbackManagerForToolRun":
        from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun

        return AsyncCallbackManagerForToolRun
    if name == "CallbackManagerForRetrieverRun":
        from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

        return CallbackManagerForRetrieverRun
    if name == "AsyncCallbackManagerForRetrieverRun":
        from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun

        return AsyncCallbackManagerForRetrieverRun
    if name == "CallbackManager":
        from langchain_core.callbacks.manager import CallbackManager

        return CallbackManager
    if name == "CallbackManagerForChainGroup":
        from langchain_core.callbacks.manager import CallbackManagerForChainGroup

        return CallbackManagerForChainGroup
    if name == "AsyncCallbackManager":
        from langchain_core.callbacks.manager import AsyncCallbackManager

        return AsyncCallbackManager
    if name == "AsyncCallbackManagerForChainGroup":
        from langchain_core.callbacks.manager import AsyncCallbackManagerForChainGroup

        return AsyncCallbackManagerForChainGroup
    if name == "StdOutCallbackHandler":
        from langchain_core.callbacks.stdout import StdOutCallbackHandler

        return StdOutCallbackHandler
    if name == "StreamingStdOutCallbackHandler":
        from langchain_core.callbacks.streaming_stdout import (
            StreamingStdOutCallbackHandler,
        )

        return StreamingStdOutCallbackHandler
    if name == "FileCallbackHandler":
        from langchain_core.callbacks.file import FileCallbackHandler

        return FileCallbackHandler
    msg = f"module {__name__} has no attribute {name}"
    raise AttributeError(msg)

__all__ = [
    "dispatch_custom_event",
    "adispatch_custom_event",
    "RetrieverManagerMixin",
    "LLMManagerMixin",
    "ChainManagerMixin",
    "ToolManagerMixin",
    "Callbacks",
    "CallbackManagerMixin",
    "RunManagerMixin",
    "BaseCallbackHandler",
    "AsyncCallbackHandler",
    "BaseCallbackManager",
    "BaseRunManager",
    "RunManager",
    "ParentRunManager",
    "AsyncRunManager",
    "AsyncParentRunManager",
    "CallbackManagerForLLMRun",
    "AsyncCallbackManagerForLLMRun",
    "CallbackManagerForChainRun",
    "AsyncCallbackManagerForChainRun",
    "CallbackManagerForToolRun",
    "AsyncCallbackManagerForToolRun",
    "CallbackManagerForRetrieverRun",
    "AsyncCallbackManagerForRetrieverRun",
    "CallbackManager",
    "CallbackManagerForChainGroup",
    "AsyncCallbackManager",
    "AsyncCallbackManagerForChainGroup",
    "StdOutCallbackHandler",
    "StreamingStdOutCallbackHandler",
    "FileCallbackHandler",
]
