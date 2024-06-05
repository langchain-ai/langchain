from __future__ import annotations

import importlib
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    Union,
    overload,
)

from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import Literal


async def aenumerate(
    iterable: AsyncIterator[Any], start: int = 0
) -> AsyncIterator[tuple[int, Any]]:
    """Async version of enumerate function."""
    i = start
    async for x in iterable:
        yield i, x
        i += 1


class IndexableBaseModel(BaseModel):
    """Allows a BaseModel to return its fields by string variable indexing."""

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class Choice(IndexableBaseModel):
    """Choice."""

    message: dict


class ChatCompletions(IndexableBaseModel):
    """Chat completions."""

    choices: List[Choice]


class ChoiceChunk(IndexableBaseModel):
    """Choice chunk."""

    delta: dict


class ChatCompletionChunk(IndexableBaseModel):
    """Chat completion chunk."""

    choices: List[ChoiceChunk]


def convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        if tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    elif role == "function":
        return FunctionMessage(content=_dict.get("content", ""), name=_dict.get("name"))  # type: ignore[arg-type]
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id"),  # type: ignore[arg-type]
            additional_kwargs=additional_kwargs,
        )
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role)  # type: ignore[arg-type]


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            # If tool calls only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def convert_openai_messages(messages: Sequence[Dict[str, Any]]) -> List[BaseMessage]:
    """Convert dictionaries representing OpenAI messages to LangChain format.

    Args:
        messages: List of dictionaries representing OpenAI messages

    Returns:
        List of LangChain BaseMessage objects.
    """
    return [convert_dict_to_message(m) for m in messages]


def _convert_message_chunk(chunk: BaseMessageChunk, i: int) -> dict:
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
    return _dict


def _convert_message_chunk_to_delta(chunk: BaseMessageChunk, i: int) -> Dict[str, Any]:
    _dict = _convert_message_chunk(chunk, i)
    return {"choices": [{"delta": _dict}]}


class ChatCompletion:
    """Chat completion."""

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


def _has_assistant_message(session: ChatSession) -> bool:
    """Check if chat session has an assistant message."""
    return any([isinstance(m, AIMessage) for m in session["messages"]])


def convert_messages_for_finetuning(
    sessions: Iterable[ChatSession],
) -> List[List[dict]]:
    """Convert messages to a list of lists of dictionaries for fine-tuning.

    Args:
        sessions: The chat sessions.

    Returns:
        The list of lists of dictionaries.
    """
    return [
        [convert_message_to_dict(s) for s in session["messages"]]
        for session in sessions
        if _has_assistant_message(session)
    ]


class Completions:
    """Completions."""

    @overload
    @staticmethod
    def create(
        messages: Sequence[Dict[str, Any]],
        *,
        provider: str = "ChatOpenAI",
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> ChatCompletions:
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
    ) -> Union[ChatCompletions, Iterable]:
        models = importlib.import_module("langchain.chat_models")
        model_cls = getattr(models, provider)
        model_config = model_cls(**kwargs)
        converted_messages = convert_openai_messages(messages)
        if not stream:
            result = model_config.invoke(converted_messages)
            return ChatCompletions(
                choices=[Choice(message=convert_message_to_dict(result))]
            )
        else:
            return (
                ChatCompletionChunk(
                    choices=[ChoiceChunk(delta=_convert_message_chunk(c, i))]
                )
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
    ) -> ChatCompletions:
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
    ) -> Union[ChatCompletions, AsyncIterator]:
        models = importlib.import_module("langchain.chat_models")
        model_cls = getattr(models, provider)
        model_config = model_cls(**kwargs)
        converted_messages = convert_openai_messages(messages)
        if not stream:
            result = await model_config.ainvoke(converted_messages)
            return ChatCompletions(
                choices=[Choice(message=convert_message_to_dict(result))]
            )
        else:
            return (
                ChatCompletionChunk(
                    choices=[ChoiceChunk(delta=_convert_message_chunk(c, i))]
                )
                async for i, c in aenumerate(model_config.astream(converted_messages))
            )


class Chat:
    """Chat."""

    def __init__(self) -> None:
        self.completions = Completions()


chat = Chat()
