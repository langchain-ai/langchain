from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
)
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk

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
    return chunk.__class__.__mro__[1](
        **{k: v for k, v in chunk.__dict__.items() if k != "type"}
    )


MessageLikeRepresentation = Union[BaseMessage, Tuple[str, str], str, Dict[str, Any]]


def _create_message_from_message_type(
    message_type: str,
    content: str,
    name: Optional[str] = None,
    tool_call_id: Optional[str] = None,
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


def _convert_to_message(
    message: MessageLikeRepresentation,
) -> BaseMessage:
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
    elif isinstance(message, tuple):
        if len(message) != 2:
            raise ValueError(f"Expected 2-tuple of (role, template), got {message}")
        message_type_str, template = message
        _message = _create_message_from_message_type(message_type_str, template)
    elif isinstance(message, dict):
        msg_kwargs = message.copy()
        try:
            msg_type = msg_kwargs.pop("role")
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
