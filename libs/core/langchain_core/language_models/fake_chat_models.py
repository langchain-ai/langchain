"""Fake ChatModel for testing purposes."""

import asyncio
import re
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional, Union, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig


class FakeMessagesListChatModel(BaseChatModel):
    """Fake ChatModel for testing purposes."""

    responses: list[BaseMessage]
    """List of responses to **cycle** through in order."""
    sleep: Optional[float] = None
    """Sleep time in seconds between responses."""
    i: int = 0
    """Internally incremented after every model invocation."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-messages-list-chat-model"


class FakeListChatModelError(Exception):
    pass


class FakeListChatModel(SimpleChatModel):
    """Fake ChatModel for testing purposes."""

    responses: list[str]
    """List of responses to **cycle** through in order."""
    sleep: Optional[float] = None
    i: int = 0
    """List of responses to **cycle** through in order."""
    error_on_chunk_number: Optional[int] = None
    """Internally incremented after every model invocation."""

    @property
    def _llm_type(self) -> str:
        return "fake-list-chat-model"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Union[list[str], None] = None,
        run_manager: Union[CallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        for i_c, c in enumerate(response):
            if self.sleep is not None:
                time.sleep(self.sleep)
            if (
                self.error_on_chunk_number is not None
                and i_c == self.error_on_chunk_number
            ):
                raise FakeListChatModelError

            yield ChatGenerationChunk(message=AIMessageChunk(content=c))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Union[list[str], None] = None,
        run_manager: Union[AsyncCallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        for i_c, c in enumerate(response):
            if self.sleep is not None:
                await asyncio.sleep(self.sleep)
            if (
                self.error_on_chunk_number is not None
                and i_c == self.error_on_chunk_number
            ):
                raise FakeListChatModelError
            yield ChatGenerationChunk(message=AIMessageChunk(content=c))

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"responses": self.responses}

    # manually override batch to preserve batch ordering with no concurrency
    def batch(
        self,
        inputs: list[Any],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[BaseMessage]:
        if isinstance(config, list):
            return [self.invoke(m, c, **kwargs) for m, c in zip(inputs, config)]
        return [self.invoke(m, config, **kwargs) for m in inputs]

    async def abatch(
        self,
        inputs: list[Any],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[BaseMessage]:
        if isinstance(config, list):
            # do Not use an async iterator here because need explicit ordering
            return [await self.ainvoke(m, c, **kwargs) for m, c in zip(inputs, config)]
        # do Not use an async iterator here because need explicit ordering
        return [await self.ainvoke(m, config, **kwargs) for m in inputs]


class FakeChatModel(SimpleChatModel):
    """Fake Chat Model wrapper for testing purposes."""

    def _call(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "fake response"

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = "fake response"
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"key": "fake"}


class GenericFakeChatModel(BaseChatModel):
    """Generic fake chat model that can be used to test the chat model interface.

    * Chat model should be usable in both sync and async tests
    * Invokes on_llm_new_token to allow for testing of callback related code for new
      tokens.
    * Includes logic to break messages into message chunk to facilitate testing of
      streaming.
    """

    messages: Iterator[Union[AIMessage, str]]
    """Get an iterator over messages.

    This can be expanded to accept other types like Callables / dicts / strings
    to make the interface more generic if needed.

    Note: if you want to pass a list, you can use `iter` to convert it to an iterator.

    Please note that streaming is not implemented yet. We should try to implement it
    in the future by delegating to invoke and then breaking the resulting output
    into message chunks.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call."""
        message = next(self.messages)
        message_ = AIMessage(content=message) if isinstance(message, str) else message
        generation = ChatGeneration(message=message_)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model."""
        chat_result = self._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        if not isinstance(chat_result, ChatResult):
            msg = (
                f"Expected generate to return a ChatResult, "
                f"but got {type(chat_result)} instead."
            )
            raise ValueError(msg)  # noqa: TRY004

        message = chat_result.generations[0].message

        if not isinstance(message, AIMessage):
            msg = (
                f"Expected invoke to return an AIMessage, "
                f"but got {type(message)} instead."
            )
            raise ValueError(msg)  # noqa: TRY004

        content = message.content

        if content:
            # Use a regular expression to split on whitespace with a capture group
            # so that we can preserve the whitespace in the output.
            if not isinstance(content, str):
                msg = "Expected content to be a string."
                raise ValueError(msg)

            content_chunks = cast(list[str], re.split(r"(\s)", content))

            for token in content_chunks:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=token, id=message.id)
                )
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk

        if message.additional_kwargs:
            for key, value in message.additional_kwargs.items():
                # We should further break down the additional kwargs into chunks
                # Special case for function call
                if key == "function_call":
                    for fkey, fvalue in value.items():
                        if isinstance(fvalue, str):
                            # Break function call by `,`
                            fvalue_chunks = cast(list[str], re.split(r"(,)", fvalue))
                            for fvalue_chunk in fvalue_chunks:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        id=message.id,
                                        content="",
                                        additional_kwargs={
                                            "function_call": {fkey: fvalue_chunk}
                                        },
                                    )
                                )
                                if run_manager:
                                    run_manager.on_llm_new_token(
                                        "",
                                        chunk=chunk,  # No token for function call
                                    )
                                yield chunk
                        else:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(
                                    id=message.id,
                                    content="",
                                    additional_kwargs={"function_call": {fkey: fvalue}},
                                )
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(
                                    "",
                                    chunk=chunk,  # No token for function call
                                )
                            yield chunk
                else:
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(
                            id=message.id, content="", additional_kwargs={key: value}
                        )
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(
                            "",
                            chunk=chunk,  # No token for function call
                        )
                    yield chunk

    @property
    def _llm_type(self) -> str:
        return "generic-fake-chat-model"


class ParrotFakeChatModel(BaseChatModel):
    """Generic fake chat model that can be used to test the chat model interface.

    * Chat model should be usable in both sync and async tests
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call."""
        return ChatResult(generations=[ChatGeneration(message=messages[-1])])

    @property
    def _llm_type(self) -> str:
        return "parrot-fake-chat-model"
