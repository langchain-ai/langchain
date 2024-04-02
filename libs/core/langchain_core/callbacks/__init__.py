"""**Callback handlers** allow listening to events in LangChain.

**Class hierarchy:**

.. code-block::

    BaseCallbackHandler --> <name>CallbackHandler  # Example: AimCallbackHandler
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "AsyncCallbackHandler":
        from langchain_core.callbacks.base import AsyncCallbackHandler

        return AsyncCallbackHandler
    elif name == "BaseCallbackHandler":
        from langchain_core.callbacks.base import BaseCallbackHandler

        return BaseCallbackHandler
    elif name == "BaseCallbackManager":
        from langchain_core.callbacks.base import BaseCallbackManager

        return BaseCallbackManager
    elif name == "CallbackManagerMixin":
        from langchain_core.callbacks.base import CallbackManagerMixin

        return CallbackManagerMixin
    elif name == "Callbacks":
        from langchain_core.callbacks.base import Callbacks

        return Callbacks
    elif name == "ChainManagerMixin":
        from langchain_core.callbacks.base import ChainManagerMixin

        return ChainManagerMixin
    elif name == "LLMManagerMixin":
        from langchain_core.callbacks.base import LLMManagerMixin

        return LLMManagerMixin
    elif name == "RetrieverManagerMixin":
        from langchain_core.callbacks.base import RetrieverManagerMixin

        return RetrieverManagerMixin
    elif name == "ToolManagerMixin":
        from langchain_core.callbacks.base import ToolManagerMixin

        return ToolManagerMixin
    elif name == "AsyncCallbackManager":
        from langchain_core.callbacks.manager import AsyncCallbackManager

        return AsyncCallbackManager
    elif name == "AsyncCallbackManagerForChainGroup":
        from langchain_core.callbacks.manager import AsyncCallbackManagerForChainGroup

        return AsyncCallbackManagerForChainGroup
    elif name == "AsyncCallbackManagerForChainRun":
        from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun

        return AsyncCallbackManagerForChainRun
    elif name == "AsyncCallbackManagerForLLMRun":
        from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun

        return AsyncCallbackManagerForLLMRun
    elif name == "AsyncCallbackManagerForRetrieverRun":
        from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun

        return AsyncCallbackManagerForRetrieverRun
    elif name == "AsyncCallbackManagerForToolRun":
        from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun

        return AsyncCallbackManagerForToolRun
    elif name == "AsyncParentRunManager":
        from langchain_core.callbacks.manager import AsyncParentRunManager

        return AsyncParentRunManager
    elif name == "AsyncRunManager":
        from langchain_core.callbacks.manager import AsyncRunManager

        return AsyncRunManager
    elif name == "BaseRunManager":
        from langchain_core.callbacks.manager import BaseRunManager

        return BaseRunManager
    elif name == "CallbackManager":
        from langchain_core.callbacks.manager import CallbackManager

        return CallbackManager
    elif name == "CallbackManagerForChainGroup":
        from langchain_core.callbacks.manager import CallbackManagerForChainGroup

        return CallbackManagerForChainGroup
    elif name == "CallbackManagerForChainRun":
        from langchain_core.callbacks.manager import CallbackManagerForChainRun

        return CallbackManagerForChainRun
    elif name == "CallbackManagerForLLMRun":
        from langchain_core.callbacks.manager import CallbackManagerForLLMRun

        return CallbackManagerForLLMRun
    elif name == "CallbackManagerForRetrieverRun":
        from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

        return CallbackManagerForRetrieverRun
    elif name == "CallbackManagerForToolRun":
        from langchain_core.callbacks.manager import CallbackManagerForToolRun

        return CallbackManagerForToolRun
    elif name == "ParentRunManager":
        from langchain_core.callbacks.manager import ParentRunManager

        return ParentRunManager
    elif name == "RunManager":
        from langchain_core.callbacks.manager import RunManager

        return RunManager
    elif name == "StdOutCallbackHandler":
        from langchain_core.callbacks.stdout import StdOutCallbackHandler

        return StdOutCallbackHandler
    elif name == "StreamingStdOutCallbackHandler":
        from langchain_core.callbacks.streaming_stdout import (
            StreamingStdOutCallbackHandler,
        )

        return StreamingStdOutCallbackHandler
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
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
]
