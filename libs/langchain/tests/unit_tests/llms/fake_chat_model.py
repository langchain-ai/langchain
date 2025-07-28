"""Fake Chat Model wrapper for testing purposes."""

import re
from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional, cast

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor
from typing_extensions import override


class FakeChatModel(SimpleChatModel):
    """Fake Chat Model wrapper for testing purposes."""

    @override
    def _call(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "fake response"

    @override
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
    """A generic fake chat model that can be used to test the chat model interface.

    * Chat model should be usable in both sync and async tests
    * Invokes on_llm_new_token to allow for testing of callback related code for new
      tokens.
    * Includes logic to break messages into message chunk to facilitate testing of
      streaming.
    """

    messages: Iterator[AIMessage]
    """Get an iterator over messages.

    This can be expanded to accept other types like Callables / dicts / strings
    to make the interface more generic if needed.

    Note: if you want to pass a list, you can use `iter` to convert it to an iterator.

    Please note that streaming is not implemented yet. We should try to implement it
    in the future by delegating to invoke and then breaking the resulting output
    into message chunks.
    """

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        message = next(self.messages)
        generation = ChatGeneration(message=message)
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
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        if not isinstance(chat_result, ChatResult):
            msg = (
                f"Expected generate to return a ChatResult, "
                f"but got {type(chat_result)} instead."
            )
            raise TypeError(msg)

        message = chat_result.generations[0].message

        if not isinstance(message, AIMessage):
            msg = (
                f"Expected invoke to return an AIMessage, "
                f"but got {type(message)} instead."
            )
            raise TypeError(msg)

        content = message.content

        if content:
            # Use a regular expression to split on whitespace with a capture group
            # so that we can preserve the whitespace in the output.
            assert isinstance(content, str)
            content_chunks = cast("list[str]", re.split(r"(\s)", content))

            for token in content_chunks:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(id=message.id, content=token),
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
                            fvalue_chunks = cast("list[str]", re.split(r"(,)", fvalue))
                            for fvalue_chunk in fvalue_chunks:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        id=message.id,
                                        content="",
                                        additional_kwargs={
                                            "function_call": {fkey: fvalue_chunk},
                                        },
                                    ),
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
                                ),
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
                            additional_kwargs={key: value},
                        ),
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(
                            "",
                            chunk=chunk,  # No token for function call
                        )
                    yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream the output of the model."""
        result = await run_in_executor(
            None,
            self._stream,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        for chunk in result:
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "generic-fake-chat-model"
