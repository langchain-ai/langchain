from __future__ import annotations

import importlib
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    Sequence,
    Union,
    overload,
)

from typing_extensions import Literal

from langchain.schema.messages import (
    AIMessageChunk,
    BaseMessageChunk,
)
from langchain.utils.openai import convert_message_to_dict, convert_openai_messages


async def aenumerate(
    iterable: AsyncIterator[Any], start: int = 0
) -> AsyncIterator[tuple[int, Any]]:
    """Async version of enumerate."""
    i = start
    async for x in iterable:
        yield i, x
        i += 1


def _convert_message_chunk_to_delta(chunk: BaseMessageChunk, i: int) -> Dict[str, Any]:
    _dict: Dict[str, Any] = {}
    if isinstance(chunk, AIMessageChunk):
        if i == 0:
            # Only shows up in the first chunk
            _dict["role"] = "assistant"
        if "function_call" in chunk.additional_kwargs:
            _dict["function_call"] = chunk.additional_kwargs["function_call"]
            # If the first chunk is a function call, the content is not empty string,
            # not missing, but None.
            if i == 0:
                _dict["content"] = None
        else:
            _dict["content"] = chunk.content
    else:
        raise ValueError(f"Got unexpected streaming chunk type: {type(chunk)}")
    # This only happens at the end of streams, and OpenAI returns as empty dict
    if _dict == {"content": ""}:
        _dict = {}
    return {"choices": [{"delta": _dict}]}


class ChatCompletion:
    @overload
    @staticmethod
    def create(
        messages: Sequence[Dict[str, Any]],
        *,
        provider: str = "ChatOpenAI",
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> dict:
        ...

    @overload
    @staticmethod
    def create(
        messages: Sequence[Dict[str, Any]],
        *,
        provider: str = "ChatOpenAI",
        stream: Literal[True],
        **kwargs: Any,
    ) -> Iterable:
        ...

    @staticmethod
    def create(
        messages: Sequence[Dict[str, Any]],
        *,
        provider: str = "ChatOpenAI",
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[dict, Iterable]:
        models = importlib.import_module("langchain.chat_models")
        model_cls = getattr(models, provider)
        model_config = model_cls(**kwargs)
        converted_messages = convert_openai_messages(messages)
        if not stream:
            result = model_config.invoke(converted_messages)
            return {"choices": [{"message": convert_message_to_dict(result)}]}
        else:
            return (
                _convert_message_chunk_to_delta(c, i)
                for i, c in enumerate(model_config.stream(converted_messages))
            )

    @overload
    @staticmethod
    async def acreate(
        messages: Sequence[Dict[str, Any]],
        *,
        provider: str = "ChatOpenAI",
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> dict:
        ...

    @overload
    @staticmethod
    async def acreate(
        messages: Sequence[Dict[str, Any]],
        *,
        provider: str = "ChatOpenAI",
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator:
        ...

    @staticmethod
    async def acreate(
        messages: Sequence[Dict[str, Any]],
        *,
        provider: str = "ChatOpenAI",
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[dict, AsyncIterator]:
        models = importlib.import_module("langchain.chat_models")
        model_cls = getattr(models, provider)
        model_config = model_cls(**kwargs)
        converted_messages = convert_openai_messages(messages)
        if not stream:
            result = await model_config.ainvoke(converted_messages)
            return {"choices": [{"message": convert_message_to_dict(result)}]}
        else:
            return (
                _convert_message_chunk_to_delta(c, i)
                async for i, c in aenumerate(model_config.astream(converted_messages))
            )
