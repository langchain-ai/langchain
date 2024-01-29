from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult


class Chat__ModuleName__(BaseChatModel):
    """Chat__ModuleName__ chat model.

    Example:
        .. code-block:: python

            from __module_name__ import Chat__ModuleName__


            model = Chat__ModuleName__()
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-__package_name_short__"

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(
            message=BaseMessageChunk(content="Yield chunks", type="ai"),
        )
        yield ChatGenerationChunk(
            message=BaseMessageChunk(content=" like this!", type="ai"),
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError
