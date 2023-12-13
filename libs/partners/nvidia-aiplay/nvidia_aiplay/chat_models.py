"""Chat Model Components Derived from ChatModel/NVAIPlay"""
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult


# from langchain.chat_models.base import SimpleChatModel
# from langchain.llms import nv_aiplay


# class NVAIPlayChat(nv_aiplay.NVAIPlayBaseModel, SimpleChatModel):
#     pass


# class GeneralChat(nv_aiplay.GeneralBase, SimpleChatModel):
#     pass


# class CodeChat(nv_aiplay.CodeBase, SimpleChatModel):
#     pass


# class InstructChat(nv_aiplay.InstructBase, SimpleChatModel):
#     pass


# class SteerChat(nv_aiplay.SteerBase, SimpleChatModel):
#     pass


# class ContextChat(nv_aiplay.ContextBase, SimpleChatModel):
#     pass


# class ImageChat(nv_aiplay.ImageBase, SimpleChatModel):
#     pass


# class ChatNVAIPlay(BaseChatModel):
#     """{integration} chat model.

#     Example:
#         .. code-block:: python

#             from nvidia_aiplay import ChatNVAIPlay


#             model = ChatNVAIPlay(raise NotImplementedError)
#     """

#     @property
#     def _llm_type(self) -> str:
#         """Return type of chat model."""
#         return "chat-integration"

#     def _stream(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> Iterator[ChatGenerationChunk]:
#         raise NotImplementedError

#     async def _astream(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> AsyncIterator[ChatGenerationChunk]:
#         yield ChatGenerationChunk(
#             message=BaseMessageChunk(content="Yield chunks", type="ai"),
#         )
#         yield ChatGenerationChunk(
#             message=BaseMessageChunk(content=" like this!", type="ai"),
#         )

#     def _generate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         raise NotImplementedError

#     async def _agenerate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         raise NotImplementedError
