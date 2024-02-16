"""Friendli Chat Model."""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional

from friendli import AsyncFriendli, Friendli
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def get_role(message: BaseMessage) -> str:
    """Get the role of the message."""
    if isinstance(message, ChatMessage) or isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    raise ValueError(f"Got unknown type {message}")


class ChatFriendli(BaseChatModel):
    """Friendli Chat."""

    token: str = ""
    model: str = "mixtral-8x7b-instruct-v0-1"
    streaming: bool = False
    client: Friendli = None
    async_client: AsyncFriendli = None

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.client = Friendli(token=self.token)
        self.async_client = AsyncFriendli(token=self.token)

    @property
    def _llm_type(self) -> str:
        return "friendli-chat"

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        stream = self.client.chat.completions.create(
            messages=[
                {
                    "role": get_role(message),
                    "content": message.content,
                }
                for message in messages
            ],
            stream=True,
            model=self.model,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    run_manager.on_llm_new_token(delta)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        stream = await self.async_client.chat.completions.create(
            messages=[
                {
                    "role": get_role(message),
                    "content": message.content,
                }
                for message in messages
            ],
            stream=True,
            model=self.model,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    await run_manager.on_llm_new_token(delta)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": get_role(message),
                    "content": message.content,
                }
                for message in messages
            ],
            stream=True,
            model=self.model,
        )

        message = AIMessage(content=response.choices[0].text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        response = await self.async_client.chat.completions.create(
            messages=[
                {
                    "role": get_role(message),
                    "content": message.content,
                }
                for message in messages
            ],
            stream=True,
            model=self.model,
        )

        message = AIMessage(content=response.choices[0].text)
        return ChatResult(generations=[ChatGeneration(message=message)])
