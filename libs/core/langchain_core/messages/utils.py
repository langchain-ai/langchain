"""Module contains utility functions for working with messages.

Some examples of what you can do with these functions include:

* Convert messages to strings (serialization)
* Convert messages from dicts to Message objects (deserialization)
* Filter messages from a list of messages based on name, type or id etc.
"""

from __future__ import annotations

import base64
import inspect
import json
import re
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)

from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk
from langchain_core.messages.tool import (
    tool_call as create_tool_call,
)
from langchain_core.messages.tool import (
    tool_call_chunk as create_tool_call_chunk,
)

if TYPE_CHECKING:
    from langchain_text_splitters import TextSplitter

    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompt_values import PromptValue
    from langchain_core.runnables.base import Runnable

AnyMessage = Union[
    AIMessage,
    HumanMessage,
    ChatMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
]


def get_buffer_string(
    messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """Convert a sequence of Messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of HumanMessages.
            Default is "Human".
        ai_prefix: THe prefix to prepend to contents of AIMessages. Default is "AI".

    Returns:
        A single string concatenation of all input messages.

    Raises:
        ValueError: If an unsupported message type is encountered.

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
    elif _type == "remove":
        return RemoveMessage(**message["data"])
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
        message_type: (str) the type of the message (e.g., "human", "ai", etc.).
        content: (str) the content string.
        name: (str) the name of the message. Default is None.
        tool_call_id: (str) the tool call id. Default is None.
        tool_calls: (List[Dict[str, Any]]) the tool calls. Default is None.
        id: (str) the id of the message. Default is None.
        **additional_kwargs: (Dict[str, Any]) additional keyword arguments.

    Returns:
        a message of the appropriate type.

    Raises:
        ValueError: if the message type is not one of "human", "user", "ai",
            "assistant", "system", "function", or "tool".
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
        kwargs["tool_calls"] = []
        for tool_call in tool_calls:
            # Convert OpenAI-format tool call to LangChain format.
            if "function" in tool_call:
                args = tool_call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args, strict=False)
                kwargs["tool_calls"].append(
                    {
                        "name": tool_call["function"]["name"],
                        "args": args,
                        "id": tool_call["id"],
                        "type": "tool_call",
                    }
                )
            else:
                kwargs["tool_calls"].append(tool_call)
    if message_type in ("human", "user"):
        message: BaseMessage = HumanMessage(content=content, **kwargs)
    elif message_type in ("ai", "assistant"):
        message = AIMessage(content=content, **kwargs)
    elif message_type == "system":
        message = SystemMessage(content=content, **kwargs)
    elif message_type == "function":
        message = FunctionMessage(content=content, **kwargs)
    elif message_type == "tool":
        artifact = kwargs.get("additional_kwargs", {}).pop("artifact", None)
        message = ToolMessage(content=content, artifact=artifact, **kwargs)
    elif message_type == "remove":
        message = RemoveMessage(**kwargs)
    else:
        raise ValueError(
            f"Unexpected message type: {message_type}. Use one of 'human',"
            f" 'user', 'ai', 'assistant', 'function', 'tool', or 'system'."
        )
    return message


def _convert_to_message(
    message: MessageLikeRepresentation, *, copy: bool = False
) -> BaseMessage:
    """Instantiate a message from a variety of message formats.

    The message format can be one of the following:

    - BaseMessagePromptTemplate
    - BaseMessage
    - 2-tuple of (role string, template); e.g., ("human", "{user_input}")
    - dict: a message dict with role and content keys
    - string: shorthand for ("human", template); e.g., "{user_input}"

    Args:
        message: a representation of a message in one of the supported formats.

    Returns:
        an instance of a message or a message template.

    Raises:
        NotImplementedError: if the message type is not supported.
        ValueError: if the message dict does not contain the required keys.
    """
    if isinstance(message, BaseMessage):
        if copy:
            _message = message.__class__(**message.dict())
        else:
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
            # None msg content is not allowed
            msg_content = msg_kwargs.pop("content") or ""
        except KeyError as e:
            raise ValueError(
                f"Message dict must contain 'role' and 'content' keys, got {message}"
            ) from e
        _message = _create_message_from_message_type(
            msg_type, msg_content, **msg_kwargs
        )
    else:
        raise NotImplementedError(f"Unsupported message type: {type(message)}")

    return _message


def convert_to_messages(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
    *,
    copy: bool = False,
) -> List[BaseMessage]:
    """Convert a sequence of messages to a list of messages.

    Args:
        messages: Sequence of messages to convert.

    Returns:
        List of messages (BaseMessages).
    """
    # Import here to avoid circular imports
    from langchain_core.prompt_values import PromptValue

    if isinstance(messages, PromptValue):
        return messages.to_messages()
    return [_convert_to_message(m, copy=copy) for m in messages]


def _runnable_support(func: Callable) -> Callable:
    @overload
    def wrapped(
        messages: Literal[None] = None, **kwargs: Any
    ) -> Runnable[Sequence[MessageLikeRepresentation], List[BaseMessage]]: ...

    @overload
    def wrapped(
        messages: Sequence[MessageLikeRepresentation], **kwargs: Any
    ) -> List[BaseMessage]: ...

    def wrapped(
        messages: Union[Sequence[MessageLikeRepresentation], None] = None,
        **kwargs: Any,
    ) -> Union[
        Runnable[Sequence[MessageLikeRepresentation], List[BaseMessage]],
        List[BaseMessage],
    ]:
        from langchain_core.runnables.base import RunnableLambda

        if messages is not None:
            return func(messages, **kwargs)
        else:
            return RunnableLambda(partial(func, **kwargs), name=func.__name__)

    wrapped.__doc__ = func.__doc__
    return wrapped


@_runnable_support
def filter_messages(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
    *,
    include_names: Optional[Sequence[str]] = None,
    exclude_names: Optional[Sequence[str]] = None,
    include_types: Optional[Sequence[Union[str, Type[BaseMessage]]]] = None,
    exclude_types: Optional[Sequence[Union[str, Type[BaseMessage]]]] = None,
    include_ids: Optional[Sequence[str]] = None,
    exclude_ids: Optional[Sequence[str]] = None,
) -> List[BaseMessage]:
    """Filter messages based on name, type or id.

    Args:
        messages: Sequence Message-like objects to filter.
        include_names: Message names to include. Default is None.
        exclude_names: Messages names to exclude. Default is None.
        include_types: Message types to include. Can be specified as string names (e.g.
            "system", "human", "ai", ...) or as BaseMessage classes (e.g.
            SystemMessage, HumanMessage, AIMessage, ...). Default is None.
        exclude_types: Message types to exclude. Can be specified as string names (e.g.
            "system", "human", "ai", ...) or as BaseMessage classes (e.g.
            SystemMessage, HumanMessage, AIMessage, ...). Default is None.
        include_ids: Message IDs to include. Default is None.
        exclude_ids: Message IDs to exclude. Default is None.

    Returns:
        A list of Messages that meets at least one of the incl_* conditions and none
        of the excl_* conditions. If not incl_* conditions are specified then
        anything that is not explicitly excluded will be included.

    Raises:
        ValueError if two incompatible arguments are provided.

    Example:
        .. code-block:: python

            from langchain_core.messages import filter_messages, AIMessage, HumanMessage, SystemMessage

            messages = [
                SystemMessage("you're a good assistant."),
                HumanMessage("what's your name", id="foo", name="example_user"),
                AIMessage("steve-o", id="bar", name="example_assistant"),
                HumanMessage("what's your favorite color", id="baz",),
                AIMessage("silicon blue", id="blah",),
            ]

            filter_messages(
                messages,
                incl_names=("example_user", "example_assistant"),
                incl_types=("system",),
                excl_ids=("bar",),
            )

        .. code-block:: python

            [
                SystemMessage("you're a good assistant."),
                HumanMessage("what's your name", id="foo", name="example_user"),
            ]
    """  # noqa: E501
    messages = convert_to_messages(messages)
    filtered: List[BaseMessage] = []
    for msg in messages:
        if exclude_names and msg.name in exclude_names:
            continue
        elif exclude_types and _is_message_type(msg, exclude_types):
            continue
        elif exclude_ids and msg.id in exclude_ids:
            continue
        else:
            pass

        # default to inclusion when no inclusion criteria given.
        if not (include_types or include_ids or include_names):
            filtered.append(msg)
        elif include_names and msg.name in include_names:
            filtered.append(msg)
        elif include_types and _is_message_type(msg, include_types):
            filtered.append(msg)
        elif include_ids and msg.id in include_ids:
            filtered.append(msg)
        else:
            pass

    return filtered


@_runnable_support
def merge_message_runs(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
    *,
    chunk_separator: str = "\n",
) -> List[BaseMessage]:
    """Merge consecutive Messages of the same type.

    **NOTE**: ToolMessages are not merged, as each has a distinct tool call id that
    can't be merged.

    Args:
        messages: Sequence Message-like objects to merge.
        chunk_separator: Specify the string to be inserted between message chunks.
        Default is "\n".

    Returns:
        List of BaseMessages with consecutive runs of message types merged into single
        messages. By default, if two messages being merged both have string contents,
        the merged content is a concatenation of the two strings with a new-line separator.
        The separator inserted between message chunks can be controlled by specifying
        any string with ``chunk_separator``. If at least one of the messages has a list of
        content blocks, the merged content is a list of content blocks.

    Example:
        .. code-block:: python

            from langchain_core.messages import (
                merge_message_runs,
                AIMessage,
                HumanMessage,
                SystemMessage,
                ToolCall,
            )

            messages = [
                SystemMessage("you're a good assistant."),
                HumanMessage("what's your favorite color", id="foo",),
                HumanMessage("wait your favorite food", id="bar",),
                AIMessage(
                    "my favorite colo",
                    tool_calls=[ToolCall(name="blah_tool", args={"x": 2}, id="123", type="tool_call")],
                    id="baz",
                ),
                AIMessage(
                    [{"type": "text", "text": "my favorite dish is lasagna"}],
                    tool_calls=[ToolCall(name="blah_tool", args={"x": -10}, id="456", type="tool_call")],
                    id="blur",
                ),
            ]

            merge_message_runs(messages)

        .. code-block:: python

            [
                SystemMessage("you're a good assistant."),
                HumanMessage("what's your favorite color\\nwait your favorite food", id="foo",),
                AIMessage(
                    [
                        "my favorite colo",
                        {"type": "text", "text": "my favorite dish is lasagna"}
                    ],
                    tool_calls=[
                        ToolCall({"name": "blah_tool", "args": {"x": 2}, "id": "123", "type": "tool_call"}),
                        ToolCall({"name": "blah_tool", "args": {"x": -10}, "id": "456", "type": "tool_call"})
                    ]
                    id="baz"
                ),
            ]

    """  # noqa: E501
    if not messages:
        return []
    messages = convert_to_messages(messages)
    merged: List[BaseMessage] = []
    for msg in messages:
        curr = msg.copy(deep=True)
        last = merged.pop() if merged else None
        if not last:
            merged.append(curr)
        elif isinstance(curr, ToolMessage) or not isinstance(curr, last.__class__):
            merged.extend([last, curr])
        else:
            last_chunk = _msg_to_chunk(last)
            curr_chunk = _msg_to_chunk(curr)
            if (
                isinstance(last_chunk.content, str)
                and isinstance(curr_chunk.content, str)
                and last_chunk.content
                and curr_chunk.content
            ):
                last_chunk.content += chunk_separator
            merged.append(_chunk_to_msg(last_chunk + curr_chunk))
    return merged


# TODO: Update so validation errors (for token_counter, for example) are raised on
# init not at runtime.
@_runnable_support
def trim_messages(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
    *,
    max_tokens: int,
    token_counter: Union[
        Callable[[List[BaseMessage]], int],
        Callable[[BaseMessage], int],
        BaseLanguageModel,
    ],
    strategy: Literal["first", "last"] = "last",
    allow_partial: bool = False,
    end_on: Optional[
        Union[str, Type[BaseMessage], Sequence[Union[str, Type[BaseMessage]]]]
    ] = None,
    start_on: Optional[
        Union[str, Type[BaseMessage], Sequence[Union[str, Type[BaseMessage]]]]
    ] = None,
    include_system: bool = False,
    text_splitter: Optional[Union[Callable[[str], List[str]], TextSplitter]] = None,
) -> List[BaseMessage]:
    """Trim messages to be below a token count.

    Args:
        messages: Sequence of Message-like objects to trim.
        max_tokens: Max token count of trimmed messages.
        token_counter: Function or llm for counting tokens in a BaseMessage or a list of
            BaseMessage. If a BaseLanguageModel is passed in then
            BaseLanguageModel.get_num_tokens_from_messages() will be used.
        strategy: Strategy for trimming.
            - "first": Keep the first <= n_count tokens of the messages.
            - "last": Keep the last <= n_count tokens of the messages.
            Default is "last".
        allow_partial: Whether to split a message if only part of the message can be
            included. If ``strategy="last"`` then the last partial contents of a message
            are included. If ``strategy="first"`` then the first partial contents of a
            message are included.
            Default is False.
        end_on: The message type to end on. If specified then every message after the
            last occurrence of this type is ignored. If ``strategy=="last"`` then this
            is done before we attempt to get the last ``max_tokens``. If
            ``strategy=="first"`` then this is done after we get the first
            ``max_tokens``. Can be specified as string names (e.g. "system", "human",
            "ai", ...) or as BaseMessage classes (e.g. SystemMessage, HumanMessage,
            AIMessage, ...). Can be a single type or a list of types.
            Default is None.
        start_on: The message type to start on. Should only be specified if
            ``strategy="last"``. If specified then every message before
            the first occurrence of this type is ignored. This is done after we trim
            the initial messages to the last ``max_tokens``. Does not
            apply to a SystemMessage at index 0 if ``include_system=True``. Can be
            specified as string names (e.g. "system", "human", "ai", ...) or as
            BaseMessage classes (e.g. SystemMessage, HumanMessage, AIMessage, ...). Can
            be a single type or a list of types.
            Default is None.
        include_system: Whether to keep the SystemMessage if there is one at index 0.
            Should only be specified if ``strategy="last"``.
            Default is False.
        text_splitter: Function or ``langchain_text_splitters.TextSplitter`` for
            splitting the string contents of a message. Only used if
            ``allow_partial=True``. If ``strategy="last"`` then the last split tokens
            from a partial message will be included. if ``strategy=="first"`` then the
            first split tokens from a partial message will be included. Token splitter
            assumes that separators are kept, so that split contents can be directly
            concatenated to recreate the original text. Defaults to splitting on
            newlines.

    Returns:
        List of trimmed BaseMessages.

    Raises:
        ValueError: if two incompatible arguments are specified or an unrecognized
            ``strategy`` is specified.

    Example:
        .. code-block:: python

            from typing import List

            from langchain_core.messages import trim_messages, AIMessage, BaseMessage, HumanMessage, SystemMessage

            messages = [
                SystemMessage("This is a 4 token text. The full message is 10 tokens."),
                HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="first"),
                AIMessage(
                    [
                        {"type": "text", "text": "This is the FIRST 4 token block."},
                        {"type": "text", "text": "This is the SECOND 4 token block."},
                    ],
                    id="second",
                ),
                HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="third"),
                AIMessage("This is a 4 token text. The full message is 10 tokens.", id="fourth"),
            ]

            def dummy_token_counter(messages: List[BaseMessage]) -> int:
                # treat each message like it adds 3 default tokens at the beginning
                # of the message and at the end of the message. 3 + 4 + 3 = 10 tokens
                # per message.

                default_content_len = 4
                default_msg_prefix_len = 3
                default_msg_suffix_len = 3

                count = 0
                for msg in messages:
                    if isinstance(msg.content, str):
                        count += default_msg_prefix_len + default_content_len + default_msg_suffix_len
                    if isinstance(msg.content, list):
                        count += default_msg_prefix_len + len(msg.content) *  default_content_len + default_msg_suffix_len
                return count

        First 30 tokens, not allowing partial messages:
            .. code-block:: python

                trim_messages(messages, max_tokens=30, token_counter=dummy_token_counter, strategy="first")

            .. code-block:: python

                [
                    SystemMessage("This is a 4 token text. The full message is 10 tokens."),
                    HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="first"),
                ]

        First 30 tokens, allowing partial messages:
            .. code-block:: python

                trim_messages(
                    messages,
                    max_tokens=30,
                    token_counter=dummy_token_counter,
                    strategy="first",
                    allow_partial=True,
                )

            .. code-block:: python

                [
                    SystemMessage("This is a 4 token text. The full message is 10 tokens."),
                    HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="first"),
                    AIMessage( [{"type": "text", "text": "This is the FIRST 4 token block."}], id="second"),
                ]

        First 30 tokens, allowing partial messages, have to end on HumanMessage:
            .. code-block:: python

                trim_messages(
                    messages,
                    max_tokens=30,
                    token_counter=dummy_token_counter,
                    strategy="first"
                    allow_partial=True,
                    end_on="human",
                )

            .. code-block:: python

                [
                    SystemMessage("This is a 4 token text. The full message is 10 tokens."),
                    HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="first"),
                ]


        Last 30 tokens, including system message, not allowing partial messages:
            .. code-block:: python

                trim_messages(messages, max_tokens=30, include_system=True, token_counter=dummy_token_counter, strategy="last")

            .. code-block:: python

                [
                    SystemMessage("This is a 4 token text. The full message is 10 tokens."),
                    HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="third"),
                    AIMessage("This is a 4 token text. The full message is 10 tokens.", id="fourth"),
                ]

        Last 40 tokens, including system message, allowing partial messages:
            .. code-block:: python

                trim_messages(
                    messages,
                    max_tokens=40,
                    token_counter=dummy_token_counter,
                    strategy="last",
                    allow_partial=True,
                    include_system=True
                )

            .. code-block:: python

                [
                    SystemMessage("This is a 4 token text. The full message is 10 tokens."),
                    AIMessage(
                        [{"type": "text", "text": "This is the FIRST 4 token block."},],
                        id="second",
                    ),
                    HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="third"),
                    AIMessage("This is a 4 token text. The full message is 10 tokens.", id="fourth"),
                ]

        Last 30 tokens, including system message, allowing partial messages, end on HumanMessage:
            .. code-block:: python

                trim_messages(
                    messages,
                    max_tokens=30,
                    token_counter=dummy_token_counter,
                    strategy="last",
                    end_on="human",
                    include_system=True,
                    allow_partial=True,
                )

            .. code-block:: python

                [
                    SystemMessage("This is a 4 token text. The full message is 10 tokens."),
                    AIMessage(
                        [{"type": "text", "text": "This is the FIRST 4 token block."},],
                        id="second",
                    ),
                    HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="third"),
                ]

        Last 40 tokens, including system message, allowing partial messages, start on HumanMessage:
            .. code-block:: python

                trim_messages(
                    messages,
                    max_tokens=40,
                    token_counter=dummy_token_counter,
                    strategy="last",
                    include_system=True,
                    allow_partial=True,
                    start_on="human"
                )

            .. code-block:: python

                [
                    SystemMessage("This is a 4 token text. The full message is 10 tokens."),
                    HumanMessage("This is a 4 token text. The full message is 10 tokens.", id="third"),
                    AIMessage("This is a 4 token text. The full message is 10 tokens.", id="fourth"),
                ]
    """  # noqa: E501

    if start_on and strategy == "first":
        raise ValueError
    if include_system and strategy == "first":
        raise ValueError
    messages = convert_to_messages(messages)
    if hasattr(token_counter, "get_num_tokens_from_messages"):
        list_token_counter = token_counter.get_num_tokens_from_messages
    elif callable(token_counter):
        if (
            list(inspect.signature(token_counter).parameters.values())[0].annotation
            is BaseMessage
        ):

            def list_token_counter(messages: Sequence[BaseMessage]) -> int:
                return sum(token_counter(msg) for msg in messages)  # type: ignore[arg-type, misc]
        else:
            list_token_counter = token_counter  # type: ignore[assignment]
    else:
        raise ValueError(
            f"'token_counter' expected to be a model that implements "
            f"'get_num_tokens_from_messages()' or a function. Received object of type "
            f"{type(token_counter)}."
        )

    try:
        from langchain_text_splitters import TextSplitter
    except ImportError:
        text_splitter_fn: Optional[Callable] = cast(Optional[Callable], text_splitter)
    else:
        if isinstance(text_splitter, TextSplitter):
            text_splitter_fn = text_splitter.split_text
        else:
            text_splitter_fn = text_splitter

    text_splitter_fn = text_splitter_fn or _default_text_splitter

    if strategy == "first":
        return _first_max_tokens(
            messages,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            text_splitter=text_splitter_fn,
            partial_strategy="first" if allow_partial else None,
            end_on=end_on,
        )
    elif strategy == "last":
        return _last_max_tokens(
            messages,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            allow_partial=allow_partial,
            include_system=include_system,
            start_on=start_on,
            end_on=end_on,
            text_splitter=text_splitter_fn,
        )
    else:
        raise ValueError(
            f"Unrecognized {strategy=}. Supported strategies are 'last' and 'first'."
        )


def _runnable_generator(func: Callable) -> Callable:
    @overload
    def wrapped(
        messages: Literal[None] = None, **kwargs: Any
    ) -> Runnable[
        Union[MessageLikeRepresentation, Sequence[MessageLikeRepresentation]],
        Union[BaseMessage, List[BaseMessage]],
    ]: ...

    @overload
    def wrapped(
        messages: Sequence[Union[BaseMessage, Dict, Tuple]], **kwargs: Any
    ) -> List[BaseMessage]: ...

    @overload
    def wrapped(messages: MessageLikeRepresentation, **kwargs: Any) -> BaseMessage: ...

    def wrapped(
        messages: Union[
            MessageLikeRepresentation, Sequence[MessageLikeRepresentation], None
        ] = None,
        **kwargs: Any,
    ) -> Union[
        BaseMessage,
        List[BaseMessage],
        Runnable[
            Union[MessageLikeRepresentation, Sequence[MessageLikeRepresentation]],
            Union[BaseMessage, List[BaseMessage]],
        ],
    ]:
        from langchain_core.runnables.base import RunnableGenerator

        if messages is not None:
            return func(messages, **kwargs)
        else:

            def transform(input_: Iterator, **kwargs: Any) -> Iterator:
                block_indexes = set()
                for x in input_:
                    msg = func(x, **kwargs)
                    # Special handling for transforming an OpenAI stream to an
                    # Anthropic stream.
                    if isinstance(msg, AIMessageChunk) and isinstance(
                        msg.content, list
                    ):
                        tool_use_ct = 0
                        for block in msg.content:
                            if not isinstance(block, dict):
                                continue
                            if "index" in block:
                                block_indexes.add(block["index"])
                            elif block.get("type") == "tool_use":
                                block["index"] = max(len(block_indexes), *block_indexes)
                                msg.tool_call_chunks[tool_use_ct]["index"] = block[
                                    "index"
                                ]
                            else:
                                pass

                            if block.get("type") == "tool_use":
                                tool_use_ct += 1
                    else:
                        block_indexes.add(0)
                    yield msg

            return RunnableGenerator(partial(transform, **kwargs), name=func.__name__)

    wrapped.__doc__ = func.__doc__
    return wrapped


@_runnable_generator
def format_messages(
    messages: Union[MessageLikeRepresentation, Sequence[MessageLikeRepresentation]],
    *,
    format: Literal["langchain-openai", "langchain-anthropic"],
    text_format: Literal["string", "block"],
) -> Union[BaseMessage, List[BaseMessage]]:
    """Convert message contents into a standard format.

    Can be used imperatively (pass in messages, get out messages) or can be used
    declaratively (call without messages, use resulting Runnable in a chain).

    .. versionadded:: 0.2.37

    Args:
        messages: Message-like object or iterable of objects whose contents are
            in OpenAI, Anthropic, Bedrock Converse, or VertexAI formats.
        format: Output message format:
        
                - "langchain-openai": 
                    BaseMessages with OpenAI-style contents.
                - "langchain-anthropic": 
                    BaseMessages with Anthropic-style contents.
        text_format: How to format string or text block contents:
        
                - "string": 
                    If a message has a string content, this is left as a string. If
                    a message has content blocks that are all of type 'text', these are
                    joined with a newline to make a single string. If a message has
                    content blocks and at least one isn't of type 'text', then
                    all blocks are left as dicts.
                - "block": 
                    If a message has a string content, this is turned into a list
                    with a single content block of type 'text'. If a message has content
                    blocks these are left as is.

    Returns:
        The return type depends on the input type:
            - BaseMessage: 
                If a single message-like object is passed in, a BaseMessage is
                returned.
            - List[BaseMessage]: 
                If a sequence of message-like objects are passed in, a list
                of BaseMessages are returned.
            - Runnable: 
                If no messages are passed in, a Runnable is generated that formats
                messages (per the above) when invoked.

    .. dropdown::  Basic usage
        :open:

        .. code-block:: python

            from langchain_core.messages import (
                format_messages,
                AIMessage,
                HumanMessage,
                SystemMessage,
                ToolMessage,
            )

            messages = [
                SystemMessage([{"type": "text", "text": "foo"}]),
                {"role": "user", "content": [{"type": "text", "text": "whats in this"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,'/9j/4AAQSk'"}}]},
                AIMessage("", tool_calls=[{"name": "analyze", "args": {"baz": "buz"}, "id": "1", "type": "tool_call"}]),
                ToolMessage("foobar", tool_call_id="1", name="bar"),
                {"role": "assistant", "content": "thats nice"},
            ]
            oai_strings = format_messages(messages, format="langchain-openai", text_format="string")
            # -> [
            #     SystemMessage(content='foo'),
            #     HumanMessage(content=[{'type': 'text', 'text': 'whats in this'}, {'type': 'image_url', 'image_url': {'url': "data:image/png;base64,'/9j/4AAQSk'"}}]),
            #     AIMessage(content='', tool_calls=[{'name': 'analyze', 'args': {'baz': 'buz'}, 'id': '1', 'type': 'tool_call'}]),
            #     ToolMessage(content='foobar', name='bar', tool_call_id='1'),
            #     AIMessage(content='thats nice')
            # ]

            anthropic_blocks = format_messages(messages, format="langchain-anthropic", text_format="block")
            # -> [
            #     SystemMessage(content=[{'type': 'text', 'text': 'foo'}]),
            #     HumanMessage(content=[{'type': 'text', 'text': 'whats in this'}, {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': "'/9j/4AAQSk'"}}]),
            #     AIMessage(content=[{'type': 'tool_use', 'input': {'baz': 'buz'}, 'id': '1', 'name': 'analyze'}], tool_calls=[{'name': 'analyze', 'args': {'baz': 'buz'}, 'id': '1', 'type': 'tool_call'}]),
            #     HumanMessage(content=[{'type': 'tool_result', 'content': 'foobar', 'tool_use_id': '1', 'is_error': False}]),
            #     AIMessage(content=[{'type': 'text', 'text': 'thats nice'}])
            # ]

    .. dropdown::  Chaining
        :open:

        .. code-block:: python

            from langchain_core.messages import format_messages
            from langchain.chat_models import init_chat_model

            formatter = format_messages(format="langchain-openai", text_format="string")
            llm = init_chat_model() | formatter

            llm.invoke(
                [{"role": "user", "content": "how are you"}],
                config={"model": "gpt-4o"},
            )
            # -> AIMessage(["I am good..."], ...)

            llm.invoke(
                [{"role": "user", "content": "whats your name"}],
                config={"model": "claude-3-5-sonnet-20240620"}
            )
            # -> AIMessage(["My name is...], ...)

    .. dropdown:: Streaming
        :open:
        
        .. code-block:: python

            from langchain_core.messages import format_messages
            from langchain.chat_models import init_chat_model

            formatter = format_messages(format="langchain-openai", text_format="string")
            
            def multiply(a: int, b: int) -> int:
                '''Return product of a and b.'''
                return a * b

            llm_with_tools = init_chat_model().bind_tools([multiply]) | formatter

            for chunk in llm_with_tools.stream(
                    "what's 5 times 2", config={"model": "claude-3-5-sonnet-20240620"}
                ):
                print(chunk)
            # -> AIMessageChunk(content='', id='run-6...', usage_metadata={'input_tokens': 370, 'output_tokens': 0, 'total_tokens': 370}),
            # AIMessageChunk(content='Certainly', id='run-6...'),
            # AIMessageChunk(content='! To', id='run-6...'),
            # AIMessageChunk(content=' calculate', id='run-6...'),
            # AIMessageChunk(content=' 5 times ', id='run-6...'),
            # AIMessageChunk(content='2, we can use', id='run-6...'),
            # AIMessageChunk(content=' the "', id='run-6...'),
            # AIMessageChunk(content='multiply" function that', id='run-6...'),
            # AIMessageChunk(content="'s", id='run-6...'),
            # AIMessageChunk(content=' available to', id='run-6...'),
            # AIMessageChunk(content=' us.', id='run-6...'),
            # AIMessageChunk(content=' Let', id='run-6...'),
            # AIMessageChunk(content="'s use", id='run-6...'),
            # AIMessageChunk(content=' this tool', id='run-6...'),
            # AIMessageChunk(content=' to', id='run-6...'),
            # AIMessageChunk(content=' get', id='run-6...'),
            # AIMessageChunk(content=' the result.', id='run-6...'),
            # AIMessageChunk(content='', id='run-6...', tool_calls=[{'name': 'multiply', 'args': {}, 'id': 'toolu_0...', 'type': 'tool_call'}], tool_call_chunks=[{'name': 'multiply', 'args': '', 'id': 'toolu_0...', 'index': 1, 'type': 'tool_call_chunk'}]),
            # AIMessageChunk(content='', id='run-6...', tool_calls=[{'name': '', 'args': {}, 'id': None, 'type': 'tool_call'}], tool_call_chunks=[{'name': None, 'args': '', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]),
            # AIMessageChunk(content='', id='run-6...', tool_calls=[{'name': '', 'args': {'a': 5}, 'id': None, 'type': 'tool_call'}], tool_call_chunks=[{'name': None, 'args': '{"a": 5', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]),
            # AIMessageChunk(content='', id='run-6...', invalid_tool_calls=[{'name': None, 'args': ', "b": 2}', 'id': None, 'error': None, 'type': 'invalid_tool_call'}], tool_call_chunks=[{'name': None, 'args': ', "b": 2}', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]),
            # AIMessageChunk(content='', response_metadata={'stop_reason': 'tool_use', 'stop_sequence': None}, id='run-6...', usage_metadata={'input_tokens': 0, 'output_tokens': 104, 'total_tokens': 104})

    """  # noqa: E501
    if is_single := isinstance(messages, (BaseMessage, dict)):
        messages = [messages]
    messages = convert_to_messages(messages, copy=True)
    if format.lower().replace("_", "-") == "langchain-openai":
        formatted = _format_messages_openai(messages, text_format=text_format)
    elif format.lower().replace("_", "-") == "langchain-anthropic":
        formatted = _format_messages_anthropic(messages, text_format=text_format)
    else:
        raise ValueError(
            f"Unrecognized {format=}. Expected one of ('langchain-openai', "
            f"'langchain-anthropic')."
        )
    if is_single:
        return formatted[0]
    else:
        return formatted


def _format_messages_openai(
    messages: Sequence[BaseMessage], *, text_format: Literal["string", "block"]
) -> List[BaseMessage]:
    """Mutates messages so their contents match OpenAI messages API."""
    updated_messages: list = []
    for i, message in enumerate(messages):
        tool_messages: list = []
        if not message.content:
            message.content = "" if text_format == "string" else []
        elif isinstance(message.content, str):
            if text_format == "string":
                pass
            else:
                message.content = [{"type": "text", "text": message.content}]
        else:
            if text_format == "string" and all(
                isinstance(block, str) or block.get("type") == "text"
                for block in message.content
            ):
                message.content = "\n".join(
                    block if isinstance(block, str) else block["text"]
                    for block in message.content
                )
            else:
                content: List[dict] = []
                for j, block in enumerate(message.content):
                    # OpenAI format
                    if isinstance(block, str):
                        content.append({"type": "text", "text": block})
                    elif block.get("type") == "text":
                        if missing := [k for k in ("text",) if k not in block]:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'text' "
                                f"but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                        content.append({"type": block["type"], "text": block["text"]})
                    elif block.get("type") == "image_url":
                        if missing := [k for k in ("image_url",) if k not in block]:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'image_url' "
                                f"but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                        content.append(
                            {"type": "image_url", "image_url": block["image_url"]}
                        )
                    # Anthropic and Bedrock converse format
                    elif (block.get("type") == "image") or "image" in block:
                        # Anthropic
                        if source := block.get("source"):
                            if missing := [
                                k
                                for k in ("media_type", "type", "data")
                                if k not in source
                            ]:
                                raise ValueError(
                                    f"Unrecognized content block at "
                                    f"messages[{i}].content[{j}] has 'type': 'image' "
                                    f"but 'source' is missing expected key(s) "
                                    f"{missing}. Full content block:\n\n{block}"
                                )
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (
                                            f"data:{source['media_type']};"
                                            f"{source['type']},{source['data']}"
                                        )
                                    },
                                }
                            )
                        # Bedrock converse
                        elif image := block.get("image"):
                            if missing := [
                                k for k in ("source", "format") if k not in image
                            ]:
                                raise ValueError(
                                    f"Unrecognized content block at "
                                    f"messages[{i}].content[{j}] has key 'image', "
                                    f"but 'image' is missing expected key(s) "
                                    f"{missing}. Full content block:\n\n{block}"
                                )
                            b64_image = _bytes_to_b64_str(image["source"]["bytes"])
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (
                                            f"data:image/{image['format']};"
                                            f"base64,{b64_image}"
                                        )
                                    },
                                }
                            )
                        else:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'image' "
                                f"but does not have a 'source' or 'image' key. Full "
                                f"content block:\n\n{block}"
                            )
                    elif block.get("type") == "tool_use":
                        if not isinstance(message, AIMessageChunk):
                            if missing := [
                                k for k in ("id", "name", "input") if k not in block
                            ]:
                                raise ValueError(
                                    f"Unrecognized content block at "
                                    f"messages[{i}].content[{j}] has 'type': "
                                    f"'tool_use', but is missing expected key(s) "
                                    f"{missing}. Full content block:\n\n{block}"
                                )
                            if not any(
                                tool_call["id"] == block["id"]
                                for tool_call in cast(AIMessage, message).tool_calls
                            ):
                                cast(AIMessage, message).tool_calls.append(
                                    create_tool_call(
                                        name=block["name"],
                                        id=block["id"],
                                        args=block["input"],
                                    )
                                )
                        else:
                            if not message.tool_call_chunks:
                                message.tool_call_chunks = [
                                    create_tool_call_chunk(
                                        id=block.get("id"),
                                        index=block.get("index"),
                                        args=block.get("partial_json"),
                                        name=block.get("name"),
                                    )
                                ]
                    elif block.get("type") == "tool_result":
                        if missing := [
                            k for k in ("content", "tool_use_id") if k not in block
                        ]:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'tool_result', but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                        tool_message = ToolMessage(
                            block["content"],
                            tool_call_id=block["tool_use_id"],
                            status="error" if block.get("is_error") else "success",
                        )
                        # Recurse to make sure tool message contents are OpenAI format.
                        tool_messages.extend(
                            _format_messages_openai(
                                [tool_message], text_format=text_format
                            )
                        )
                    elif (block.get("type") == "json") or "json" in block:
                        if "json" not in block:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'json' "
                                f"but does not have a 'json' key. Full "
                                f"content block:\n\n{block}"
                            )
                        content.append(
                            {"type": "text", "text": json.dumps(block["json"])}
                        )
                    elif (
                        block.get("type") == "guard_content"
                    ) or "guard_content" in block:
                        if (
                            "guard_content" not in block
                            or "text" not in block["guard_content"]
                        ):
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'guard_content' but does not have a "
                                f"messages[{i}].content[{j}]['guard_content']['text'] "
                                f"key. Full content block:\n\n{block}"
                            )
                        text = block["guard_content"]["text"]
                        if isinstance(text, dict):
                            text = text["text"]
                        content.append({"type": "text", "text": text})
                    # VertexAI format
                    elif block.get("type") == "media":
                        if missing := [
                            k for k in ("mime_type", "data") if k not in block
                        ]:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'media' but does not have key(s) {missing}. Full "
                                f"content block:\n\n{block}"
                            )
                        if "image" not in block["mime_type"]:
                            raise ValueError(
                                f"OpenAI messages can only support text and image data."
                                f" Received content block with media of type:"
                                f" {block['mime_type']}"
                            )
                        b64_image = _bytes_to_b64_str(block["data"])
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:{block['mime_type']};base64,{b64_image}"
                                    )
                                },
                            }
                        )
                    else:
                        raise ValueError(
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] does not match OpenAI, "
                            f"Anthropic, Bedrock Converse, or VertexAI format. Full "
                            f"content block:\n\n{block}"
                        )
                if text_format == "string" and not any(
                    block["type"] != "text" for block in content
                ):
                    message.content = "\n".join(block["text"] for block in content)
                else:
                    message.content = content  # type: ignore[assignment]
        updated_messages.extend([message, *tool_messages])
    return updated_messages


_OPTIONAL_ANTHROPIC_KEYS = ("cache_control", "is_error", "index")


def _format_messages_anthropic(
    messages: Sequence[BaseMessage], *, text_format: Literal["string", "block"]
) -> List[BaseMessage]:
    """Mutates messages so their contents match Anthropic messages API."""
    updated_messages: List = []
    for i, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            tool_result_block = {
                "type": "tool_result",
                "content": message.content,
                "tool_use_id": message.tool_call_id,
                "is_error": message.status == "error",
            }
            if updated_messages and isinstance(updated_messages[-1], HumanMessage):
                if isinstance(updated_messages[-1].content, str):
                    updated_messages[-1].content = [
                        {"type": "text", "text": updated_messages[-1].content}
                    ]
                updated_messages[-1].content.append(tool_result_block)
            else:
                updated_messages.append(HumanMessage([tool_result_block]))
            continue
        elif not message.content:
            message.content = "" if text_format == "string" else []
        elif isinstance(message.content, str):
            if text_format == "string":
                pass
            else:
                text_block: dict = {"type": "text", "text": message.content}
                if isinstance(message, AIMessageChunk):
                    text_block["index"] = 0
                message.content = [text_block]
        else:
            if text_format == "string" and all(
                isinstance(block, str)
                or (block.get("type") == "text" and "cache_control" not in block)
                for block in message.content
            ):
                message.content = "\n".join(
                    block if isinstance(block, str) else block["text"]
                    for block in message.content
                )
            else:
                content = []
                for j, block in enumerate(message.content):
                    if isinstance(block, dict):
                        block_extra = {
                            k: block[k] for k in _OPTIONAL_ANTHROPIC_KEYS if k in block
                        }
                    else:
                        block_extra = {}

                    # OpenAI format
                    if isinstance(block, str):
                        text_block = {"type": "text", "text": block}
                        if isinstance(message, AIMessageChunk):
                            text_block["index"] = 0
                        content.append(text_block)
                    elif block.get("type") == "text":
                        if missing := [k for k in ("text",) if k not in block]:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'text' "
                                f"but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                        content.append(
                            {"type": "text", "text": block["text"], **block_extra}
                        )
                    elif block.get("type") == "image_url":
                        if missing := [k for k in ("image_url",) if k not in block]:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'image_url' "
                                f"but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                        content.append(
                            {**_openai_image_to_anthropic(block), **block_extra}
                        )
                    # Anthropic and Bedrock converse format
                    elif (block.get("type") == "image") or "image" in block:
                        # Anthropic
                        if source := block.get("source"):
                            if missing := [
                                k
                                for k in ("media_type", "type", "data")
                                if k not in source
                            ]:
                                raise ValueError(
                                    f"Unrecognized content block at "
                                    f"messages[{i}].content[{j}] has 'type': 'image' "
                                    f"but 'source' is missing expected key(s) "
                                    f"{missing}. Full content block:\n\n{block}"
                                )
                            content.append(
                                {
                                    "type": "image",
                                    "source": block["source"],
                                    **block_extra,
                                }
                            )
                        # Bedrock converse
                        elif image := block.get("image"):
                            if missing := [
                                k for k in ("source", "format") if k not in image
                            ]:
                                raise ValueError(
                                    f"Unrecognized content block at "
                                    f"messages[{i}].content[{j}] has key 'image', "
                                    f"but 'image' is missing expected key(s) "
                                    f"{missing}. Full content block:\n\n{block}"
                                )
                            content.append(
                                {
                                    **_bedrock_converse_image_to_anthropic(
                                        block["image"]
                                    ),
                                    **block_extra,
                                }
                            )
                        else:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'image' "
                                f"but does not have a 'source' or 'image' key. Full "
                                f"content block:\n\n{block}"
                            )
                    elif block.get("type") == "tool_use":
                        if not isinstance(message, AIMessageChunk):
                            if missing := [
                                k for k in ("id", "name", "input") if k not in block
                            ]:
                                raise ValueError(
                                    f"Unrecognized content block at "
                                    f"messages[{i}].content[{j}] has 'type': "
                                    f"'tool_use', "
                                    f"but is missing expected key(s) "
                                    f"{missing}. Full content block:\n\n{block}"
                                )
                            content.append(
                                {
                                    "type": "tool_use",
                                    "name": block["name"],
                                    "id": block["id"],
                                    "input": block["input"],
                                    **block_extra,
                                }
                            )
                            if not any(
                                tool_call["id"] == block["id"]
                                for tool_call in cast(AIMessage, message).tool_calls
                            ):
                                cast(AIMessage, message).tool_calls.append(
                                    create_tool_call(
                                        name=block["name"],
                                        id=block["id"],
                                        args=block["input"],
                                    )
                                )
                        else:
                            if (
                                not any(k in block for k in ("input", "partial_json"))
                                or "index" not in block
                            ):
                                raise ValueError(
                                    f"Unrecognized content block at "
                                    f"message_chunks[{i}].content[{j}] has "
                                    f"'type': 'tool_use', "
                                    f"but is does not have either an 'input' or "
                                    f"'partial_json' and an 'index' key. Full content "
                                    f"block:\n\n{block}"
                                )
                            content.append(
                                {
                                    "type": "tool_use",
                                    "index": block["index"],
                                    **{
                                        k: block[k]
                                        for k in ("name", "input", "partial_json", "id")
                                        if k in block
                                    },
                                }
                            )
                            if not message.tool_call_chunks:
                                message.tool_call_chunks = [
                                    create_tool_call_chunk(
                                        name=block.get("name"),
                                        id=block.get("id"),
                                        index=block["index"],
                                        args=block["partial_json"]
                                        if "partial_json" in block
                                        else block["input"],
                                    )
                                ]
                    elif block.get("type") == "tool_result":
                        if missing := [
                            k for k in ("content", "tool_use_id") if k not in block
                        ]:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'tool_result', but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                        content.append(
                            {
                                "type": "tool_result",
                                "content": block["content"],
                                "tool_use_id": block["tool_use_id"],
                                **block_extra,
                            }
                        )
                    elif (block.get("type") == "json") or "json" in block:
                        if "json" not in block:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'json' "
                                f"but does not have a 'json' key. Full "
                                f"content block:\n\n{block}"
                            )
                        content.append(
                            {
                                "type": "text",
                                "text": json.dumps(block["json"]),
                                **block_extra,
                            }
                        )
                    elif (
                        block.get("type") == "guard_content"
                    ) or "guard_content" in block:
                        if (
                            "guard_content" not in block
                            or "text" not in block["guard_content"]
                        ):
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'guard_content' but does not have a "
                                f"messages[{i}].content[{j}]['guard_content']['text'] "
                                f"key. Full content block:\n\n{block}"
                            )
                        text = block["guard_content"]["text"]
                        if isinstance(text, dict):
                            text = text["text"]
                        content.append({"type": "text", "text": text, **block_extra})
                    # VertexAI format
                    elif block.get("type") == "media":
                        if missing := [
                            k for k in ("mime_type", "data") if k not in block
                        ]:
                            raise ValueError(
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'media' but does not have key(s) {missing}. Full "
                                f"content block:\n\n{block}"
                            )
                        if "image" not in block["mime_type"]:
                            raise ValueError(
                                f"Anthropic messages can only support text and image "
                                f"data. Received content block with media of type: "
                                f"{block['mime_type']}"
                            )
                        content.append(
                            {**_vertexai_image_to_anthropic(block), **block_extra}
                        )
                    else:
                        raise ValueError(
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] does not match OpenAI, "
                            f"Anthropic, Bedrock Converse, or VertexAI format. Full "
                            f"content block:\n\n{block}"
                        )
                message.content = content  # type: ignore[assignment]

        if isinstance(message, AIMessageChunk) and message.tool_call_chunks:
            if isinstance(message.content, str):
                if message.content:
                    message.content = [
                        {"type": "text", "text": message.content, "index": 0}
                    ]
                else:
                    message.content = []
            if not any(
                cast(dict, block).get("type") == "tool_use" for block in message.content
            ):
                tool_use_blocks = [
                    # Note: we intentionally omit index so that it can be set by the
                    # stream handler, which can count how many blocks
                    # have been seen in preceding chunks
                    {
                        "type": "tool_use",
                        "partial_json": tc_chunk["args"],
                        "id": tc_chunk["id"],
                        "name": tc_chunk["name"],
                    }
                    for i, tc_chunk in enumerate(message.tool_call_chunks)
                ]
                tool_use_blocks = [
                    {k: v for k, v in tu_block.items() if v is not None}
                    for tu_block in tool_use_blocks
                ]
                message.content.extend(tool_use_blocks)
        elif isinstance(message, AIMessage) and message.tool_calls:
            if isinstance(message.content, str):
                message.content = [{"type": "text", "text": message.content}]
            for tool_call in message.tool_calls:
                if not any(
                    block.get("type") == "tool_use"
                    and block.get("id") == tool_call["id"]
                    for block in cast(List[dict], message.content)
                ):
                    message.content.append(
                        {
                            "type": "tool_use",
                            "input": tool_call["args"],
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                        }
                    )
        updated_messages.append(message)
    return merge_message_runs(updated_messages)


def _first_max_tokens(
    messages: Sequence[BaseMessage],
    *,
    max_tokens: int,
    token_counter: Callable[[List[BaseMessage]], int],
    text_splitter: Callable[[str], List[str]],
    partial_strategy: Optional[Literal["first", "last"]] = None,
    end_on: Optional[
        Union[str, Type[BaseMessage], Sequence[Union[str, Type[BaseMessage]]]]
    ] = None,
) -> List[BaseMessage]:
    messages = list(messages)
    idx = 0
    for i in range(len(messages)):
        if token_counter(messages[:-i] if i else messages) <= max_tokens:
            idx = len(messages) - i
            break

    if idx < len(messages) - 1 and partial_strategy:
        included_partial = False
        if isinstance(messages[idx].content, list):
            excluded = messages[idx].copy(deep=True)
            num_block = len(excluded.content)
            if partial_strategy == "last":
                excluded.content = list(reversed(excluded.content))
            for _ in range(1, num_block):
                excluded.content = excluded.content[:-1]
                if token_counter(messages[:idx] + [excluded]) <= max_tokens:
                    messages = messages[:idx] + [excluded]
                    idx += 1
                    included_partial = True
                    break
            if included_partial and partial_strategy == "last":
                excluded.content = list(reversed(excluded.content))
        if not included_partial:
            excluded = messages[idx].copy(deep=True)
            if isinstance(excluded.content, list) and any(
                isinstance(block, str) or block["type"] == "text"
                for block in messages[idx].content
            ):
                text_block = next(
                    block
                    for block in messages[idx].content
                    if isinstance(block, str) or block["type"] == "text"
                )
                text = (
                    text_block["text"] if isinstance(text_block, dict) else text_block
                )
            elif isinstance(excluded.content, str):
                text = excluded.content
            else:
                text = None
            if text:
                split_texts = text_splitter(text)
                num_splits = len(split_texts)
                if partial_strategy == "last":
                    split_texts = list(reversed(split_texts))
                for _ in range(num_splits - 1):
                    split_texts.pop()
                    excluded.content = "".join(split_texts)
                    if token_counter(messages[:idx] + [excluded]) <= max_tokens:
                        if partial_strategy == "last":
                            excluded.content = "".join(reversed(split_texts))
                        messages = messages[:idx] + [excluded]
                        idx += 1
                        break

    if end_on:
        while idx > 0 and not _is_message_type(messages[idx - 1], end_on):
            idx -= 1

    return messages[:idx]


def _last_max_tokens(
    messages: Sequence[BaseMessage],
    *,
    max_tokens: int,
    token_counter: Callable[[List[BaseMessage]], int],
    text_splitter: Callable[[str], List[str]],
    allow_partial: bool = False,
    include_system: bool = False,
    start_on: Optional[
        Union[str, Type[BaseMessage], Sequence[Union[str, Type[BaseMessage]]]]
    ] = None,
    end_on: Optional[
        Union[str, Type[BaseMessage], Sequence[Union[str, Type[BaseMessage]]]]
    ] = None,
) -> List[BaseMessage]:
    messages = list(messages)
    if end_on:
        while messages and not _is_message_type(messages[-1], end_on):
            messages.pop()
    swapped_system = include_system and isinstance(messages[0], SystemMessage)
    if swapped_system:
        reversed_ = messages[:1] + messages[1:][::-1]
    else:
        reversed_ = messages[::-1]

    reversed_ = _first_max_tokens(
        reversed_,
        max_tokens=max_tokens,
        token_counter=token_counter,
        text_splitter=text_splitter,
        partial_strategy="last" if allow_partial else None,
        end_on=start_on,
    )
    if swapped_system:
        return reversed_[:1] + reversed_[1:][::-1]
    else:
        return reversed_[::-1]


_MSG_CHUNK_MAP: Dict[Type[BaseMessage], Type[BaseMessageChunk]] = {
    HumanMessage: HumanMessageChunk,
    AIMessage: AIMessageChunk,
    SystemMessage: SystemMessageChunk,
    ToolMessage: ToolMessageChunk,
    FunctionMessage: FunctionMessageChunk,
    ChatMessage: ChatMessageChunk,
}
_CHUNK_MSG_MAP = {v: k for k, v in _MSG_CHUNK_MAP.items()}


def _msg_to_chunk(message: BaseMessage) -> BaseMessageChunk:
    if message.__class__ in _MSG_CHUNK_MAP:
        return _MSG_CHUNK_MAP[message.__class__](**message.dict(exclude={"type"}))

    for msg_cls, chunk_cls in _MSG_CHUNK_MAP.items():
        if isinstance(message, msg_cls):
            return chunk_cls(**message.dict(exclude={"type"}))

    raise ValueError(
        f"Unrecognized message class {message.__class__}. Supported classes are "
        f"{list(_MSG_CHUNK_MAP.keys())}"
    )


def _chunk_to_msg(chunk: BaseMessageChunk) -> BaseMessage:
    if chunk.__class__ in _CHUNK_MSG_MAP:
        return _CHUNK_MSG_MAP[chunk.__class__](
            **chunk.dict(exclude={"type", "tool_call_chunks"})
        )
    for chunk_cls, msg_cls in _CHUNK_MSG_MAP.items():
        if isinstance(chunk, chunk_cls):
            return msg_cls(**chunk.dict(exclude={"type", "tool_call_chunks"}))

    raise ValueError(
        f"Unrecognized message chunk class {chunk.__class__}. Supported classes are "
        f"{list(_CHUNK_MSG_MAP.keys())}"
    )


def _default_text_splitter(text: str) -> List[str]:
    splits = text.split("\n")
    return [s + "\n" for s in splits[:-1]] + splits[-1:]


def _is_message_type(
    message: BaseMessage,
    type_: Union[str, Type[BaseMessage], Sequence[Union[str, Type[BaseMessage]]]],
) -> bool:
    types = [type_] if isinstance(type_, (str, type)) else type_
    types_str = [t for t in types if isinstance(t, str)]
    types_types = tuple(t for t in types if isinstance(t, type))

    return message.type in types_str or isinstance(message, types_types)


def _bytes_to_b64_str(bytes_: bytes) -> str:
    return base64.b64encode(bytes_).decode("utf-8")


def _openai_image_to_anthropic(image: dict) -> Dict:
    """
    Formats an image of format data:image/jpeg;base64,{b64_string}
    to a dict for anthropic api

    {
      "type": "base64",
      "media_type": "image/jpeg",
      "data": "/9j/4AAQSkZJRg...",
    }

    And throws an error if it's not a b64 image
    """
    regex = r"^data:(?P<media_type>image/.+);base64,(?P<data>.+)$"
    match = re.match(regex, image["image_url"]["url"])
    if match is None:
        raise ValueError(
            "Anthropic only supports base64-encoded images currently."
            " Example: data:image/png;base64,'/9j/4AAQSk'..."
        )
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": match.group("media_type"),
            "data": match.group("data"),
        },
    }


def _bedrock_converse_image_to_anthropic(image: dict) -> dict:
    return {
        "type": "image",
        "source": {
            "media_type": f"image/{image['format']}",
            "type": "base64",
            "data": _bytes_to_b64_str(image["source"]["bytes"]),
        },
    }


def _vertexai_image_to_anthropic(image: dict) -> dict:
    return {
        "type": "image",
        "source": {
            "media_type": image["mime_type"],
            "type": "base64",
            "data": _bytes_to_b64_str(image["data"]),
        },
    }
