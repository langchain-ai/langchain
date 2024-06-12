from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk
from langchain_core.prompts import BasePromptTemplate

if TYPE_CHECKING:
    from langchain_core.language_models.base import LanguageModelLike

AnyMessage = Union[
    AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage, ToolMessage
]


def get_buffer_string(
    messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """Convert a sequence of Messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of HumanMessages.
        ai_prefix: THe prefix to prepend to contents of AIMessages.

    Returns:
        A single string concatenation of all input messages.

    Example:
        .. code-block:: python

            from langchain_core import AIMessage, HumanMessage

            messages = [
                HumanMessage(content="Hi, how are you?"),
                AIMessage(content="Good, how are you?"),
            ]
            get_buffer_string(messages)
            # -> "Human: Hi, how are you?\nAI: Good, how are you?"
    """
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, FunctionMessage):
            role = "Function"
        elif isinstance(m, ToolMessage):
            role = "Tool"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        message = f"{role}: {m.content}"
        if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
            message += f"{m.additional_kwargs['function_call']}"
        string_messages.append(message)

    return "\n".join(string_messages)


def _message_from_dict(message: dict) -> BaseMessage:
    _type = message["type"]
    if _type == "human":
        return HumanMessage(**message["data"])
    elif _type == "ai":
        return AIMessage(**message["data"])
    elif _type == "system":
        return SystemMessage(**message["data"])
    elif _type == "chat":
        return ChatMessage(**message["data"])
    elif _type == "function":
        return FunctionMessage(**message["data"])
    elif _type == "tool":
        return ToolMessage(**message["data"])
    elif _type == "AIMessageChunk":
        return AIMessageChunk(**message["data"])
    elif _type == "HumanMessageChunk":
        return HumanMessageChunk(**message["data"])
    elif _type == "FunctionMessageChunk":
        return FunctionMessageChunk(**message["data"])
    elif _type == "ToolMessageChunk":
        return ToolMessageChunk(**message["data"])
    elif _type == "SystemMessageChunk":
        return SystemMessageChunk(**message["data"])
    elif _type == "ChatMessageChunk":
        return ChatMessageChunk(**message["data"])
    else:
        raise ValueError(f"Got unexpected message type: {_type}")


def messages_from_dict(messages: Sequence[dict]) -> List[BaseMessage]:
    """Convert a sequence of messages from dicts to Message objects.

    Args:
        messages: Sequence of messages (as dicts) to convert.

    Returns:
        List of messages (BaseMessages).
    """
    return [_message_from_dict(m) for m in messages]


def message_chunk_to_message(chunk: BaseMessageChunk) -> BaseMessage:
    """Convert a message chunk to a message.

    Args:
        chunk: Message chunk to convert.

    Returns:
        Message.
    """
    if not isinstance(chunk, BaseMessageChunk):
        return chunk
    # chunk classes always have the equivalent non-chunk class as their first parent
    ignore_keys = ["type"]
    if isinstance(chunk, AIMessageChunk):
        ignore_keys.append("tool_call_chunks")
    return chunk.__class__.__mro__[1](
        **{k: v for k, v in chunk.__dict__.items() if k not in ignore_keys}
    )


MessageLikeRepresentation = Union[
    BaseMessage, List[str], Tuple[str, str], str, Dict[str, Any]
]


def _create_message_from_message_type(
    message_type: str,
    content: str,
    name: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    id: Optional[str] = None,
    **additional_kwargs: Any,
) -> BaseMessage:
    """Create a message from a message type and content string.

    Args:
        message_type: str the type of the message (e.g., "human", "ai", etc.)
        content: str the content string.

    Returns:
        a message of the appropriate type.
    """
    kwargs: Dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if tool_call_id is not None:
        kwargs["tool_call_id"] = tool_call_id
    if additional_kwargs:
        kwargs["additional_kwargs"] = additional_kwargs  # type: ignore[assignment]
    if id is not None:
        kwargs["id"] = id
    if tool_calls is not None:
        kwargs["tool_calls"] = tool_calls
    if message_type in ("human", "user"):
        message: BaseMessage = HumanMessage(content=content, **kwargs)
    elif message_type in ("ai", "assistant"):
        message = AIMessage(content=content, **kwargs)
    elif message_type == "system":
        message = SystemMessage(content=content, **kwargs)
    elif message_type == "function":
        message = FunctionMessage(content=content, **kwargs)
    elif message_type == "tool":
        message = ToolMessage(content=content, **kwargs)
    else:
        raise ValueError(
            f"Unexpected message type: {message_type}. Use one of 'human',"
            f" 'user', 'ai', 'assistant', or 'system'."
        )
    return message


def _convert_to_message(message: MessageLikeRepresentation) -> BaseMessage:
    """Instantiate a message from a variety of message formats.

    The message format can be one of the following:

    - BaseMessagePromptTemplate
    - BaseMessage
    - 2-tuple of (role string, template); e.g., ("human", "{user_input}")
    - dict: a message dict with role and content keys
    - string: shorthand for ("human", template); e.g., "{user_input}"

    Args:
        message: a representation of a message in one of the supported formats

    Returns:
        an instance of a message or a message template
    """
    if isinstance(message, BaseMessage):
        _message = message
    elif isinstance(message, str):
        _message = _create_message_from_message_type("human", message)
    elif isinstance(message, Sequence) and len(message) == 2:
        # mypy doesn't realise this can't be a string given the previous branch
        message_type_str, template = message  # type: ignore[misc]
        _message = _create_message_from_message_type(message_type_str, template)
    elif isinstance(message, dict):
        msg_kwargs = message.copy()
        try:
            try:
                msg_type = msg_kwargs.pop("role")
            except KeyError:
                msg_type = msg_kwargs.pop("type")
            msg_content = msg_kwargs.pop("content")
        except KeyError:
            raise ValueError(
                f"Message dict must contain 'role' and 'content' keys, got {message}"
            )
        _message = _create_message_from_message_type(
            msg_type, msg_content, **msg_kwargs
        )
    else:
        raise NotImplementedError(f"Unsupported message type: {type(message)}")

    return _message


def convert_to_messages(
    messages: Sequence[MessageLikeRepresentation],
) -> List[BaseMessage]:
    """Convert a sequence of messages to a list of messages.

    Args:
        messages: Sequence of messages to convert.

    Returns:
        List of messages (BaseMessages).
    """
    return [_convert_to_message(m) for m in messages]


def filter_messages(
    messages: Sequence[MessageLikeRepresentation],
    *,
    incl_names: Optional[Sequence[str]] = None,
    excl_names: Optional[Sequence[str]] = None,
    incl_types: Optional[Sequence[Union[str, Type[BaseMessage]]]] = None,
    excl_types: Optional[Sequence[Union[str, Type[BaseMessage]]]] = None,
    incl_ids: Optional[Sequence[str]] = None,
    excl_ids: Optional[Sequence[str]] = None,
) -> List[BaseMessage]:
    if incl_names and excl_names:
        raise ValueError
    if incl_types and excl_types:
        raise ValueError
    if incl_ids and excl_ids:
        raise ValueError
    messages = convert_to_messages(messages)
    incl_types_str = [t for t in (incl_types or ()) if isinstance(t, str)]
    incl_types_types = tuple(t for t in (incl_types or ()) if isinstance(t, type))
    excl_types_str = [t for t in (excl_types or ()) if isinstance(t, str)]
    excl_types_types = tuple(t for t in (excl_types or ()) if isinstance(t, type))

    filtered: List[BaseMessage] = []
    for msg in messages:
        if incl_names and msg.name not in incl_names:
            continue
        elif excl_names and msg.name in excl_names:
            continue
        elif incl_types_str and msg.type not in incl_types_str:
            continue
        elif incl_types_types and not isinstance(msg, incl_types_types):
            continue
        elif excl_types_str and msg.type in excl_types_str:
            continue
        elif excl_types_types and isinstance(msg.type, excl_types_types):
            continue
        elif incl_ids and msg.id not in incl_ids:
            continue
        elif excl_ids and msg.id in excl_ids:
            continue
        else:
            filtered.append(msg)
    return filtered


def merge_message_runs(
    messages: Sequence[MessageLikeRepresentation],
) -> List[BaseMessage]:
    if not messages:
        return []
    messages = convert_to_messages(messages)
    merged = [messages[0].copy(deep=True)]
    for msg in messages[1:]:
        curr = msg.copy(deep=True)
        last = merged.pop()
        if not isinstance(curr, last.__class__):
            merged.extend([last, curr])
        else:
            merged.append(_chunk_to_msg(_msg_to_chunk(last) + _msg_to_chunk(curr)))
    return merged


def trim_messages(
    messages: Sequence[MessageLikeRepresentation],
    *,
    n_tokens: int,
    token_counter: Union[
        Callable[[Sequence[BaseMessage]], int], Callable[[BaseMessage], int]
    ],
    strategy: Literal["first", "last"] = "last",
    allow_partial: bool = False,
    end_on: Optional[Sequence[Union[str, Type[BaseMessage]]]] = None,
    start_on: Optional[Sequence[Union[str, Type[BaseMessage]]]] = None,
    keep_system: bool = False,
) -> List[BaseMessage]:
    messages = convert_to_messages(messages)
    if (
        list(inspect.signature(token_counter).parameters.values())[0].annotation
        is BaseMessage
    ):

        def list_token_counter(messages: Sequence[BaseMessage]) -> int:
            return sum(token_counter(msg) for msg in messages)
    else:
        list_token_counter = token_counter

    if strategy == "first":
        return _first_n_tokens(
            messages,
            n_tokens=n_tokens,
            token_counter=list_token_counter,
            allow_partial=allow_partial,
            end_on=end_on,
        )
    elif strategy == "last":
        return _last_n_tokens(
            messages,
            n_tokens=n_tokens,
            token_counter=list_token_counter,
            allow_partial=allow_partial,
            keep_system=keep_system,
            start_on=start_on,
        )
    else:
        raise ValueError(
            f"Unrecognized {strategy=}. Supported strategies are 'last' and 'first'."
        )


def _first_n_tokens(
    messages: Sequence[BaseMessage],
    *,
    n_tokens: int,
    token_counter: Callable[[Sequence[BaseMessage]], int],
    allow_partial: bool = False,
    end_on: Optional[Sequence[Union[str, Type[BaseMessage]]]] = None,
) -> List[BaseMessage]:
    idx = 0
    for i in range(len(messages)):
        if token_counter(messages[:-i] if i else messages) <= n_tokens:
            idx = len(messages) - i
            break

    if idx < len(messages) - 1 and allow_partial:
        excluded = messages[idx].copy(deep=True)
        if isinstance(excluded.content, list):
            ...
        # TODO: Should we just pass in tokenizer, so we can find exact cutoff?
        else:
            ...

    if isinstance(end_on, str):
        while messages[idx - 1].type != end_on and idx > 0:
            idx = idx - 1
    elif isinstance(end_on, type):
        while not isinstance(messages[idx - 1], end_on) and idx > 0:
            idx = idx - 1
    else:
        pass

    return [msg for msg in messages[:idx]]


def _last_n_tokens(
    messages: Sequence[BaseMessage],
    *,
    n_tokens: int,
    token_counter: Callable[[Sequence[BaseMessage]], int],
    allow_partial: bool = False,
    keep_system: bool = False,
    start_on: Optional[Sequence[Union[str, Type[BaseMessage]]]] = None,
) -> List[BaseMessage]:
    messages = list(messages)
    swapped_system = keep_system and isinstance(messages[0], SystemMessage)
    if swapped_system:
        reversed_ = messages[:1] + messages[1::-1]
    else:
        reversed_ = messages[::-1]

    reversed_ = _first_n_tokens(
        reversed_,
        n_tokens=n_tokens,
        token_counter=token_counter,
        allow_partial=allow_partial,
        end_on=start_on,
    )
    if swapped_system:
        return reversed_[:1] + reversed_[1::-1]
    else:
        return reversed_[::-1]


# TODO: Should this return a Runnable?
def _summarize(
    messages: Sequence[BaseMessage],
    *,
    n_tokens: int,
    token_counter: Callable[[Sequence[BaseMessage]], int],
    llm: LanguageModelLike,
    chunk_size: int = 1,
    prompt: Optional[BasePromptTemplate] = None,
) -> List[BaseMessage]:
    prompt = prompt or DEFAULT_SUMMARY_PROMPT
    ...


_MSG_CHUNK_MAP = {
    HumanMessage: HumanMessageChunk,
    AIMessage: AIMessageChunk,
    SystemMessage: SystemMessageChunk,
    ToolMessage: ToolMessageChunk,
    FunctionMessage: FunctionMessageChunk,
    ChatMessage: ChatMessageChunk,
}
_CHUNK_MSG_MAP = {v: k for k, v in _MSG_CHUNK_MAP.items()}


def _msg_to_chunk(message: AnyMessage) -> BaseMessageChunk:
    if message.__class__ in _MSG_CHUNK_MAP:
        return _MSG_CHUNK_MAP[message.__class__](**message.dict())

    for msg_cls, chunk_cls in _MSG_CHUNK_MAP.items():
        if isinstance(message, msg_cls):
            return chunk_cls(**message.dict())

    raise ValueError(
        f"Unrecognized message class {message.__class__}. Supported classes are "
        f"{list(_MSG_CHUNK_MAP.keys())}"
    )


def _chunk_to_msg(chunk: BaseMessageChunk) -> AnyMessage:
    # TODO: does this break when there are extra fields in chunk.
    if chunk.__class__ in _CHUNK_MSG_MAP:
        return _CHUNK_MSG_MAP[chunk.__class__](**chunk.dict())
    for chunk_cls, msg_cls in _CHUNK_MSG_MAP.items():
        if isinstance(chunk, chunk_cls):
            return msg_cls(**chunk.dict())

    raise ValueError(
        f"Unrecognized message chunk class {chunk.__class__}. Supported classes are "
        f"{list(_CHUNK_MSG_MAP.keys())}"
    )
