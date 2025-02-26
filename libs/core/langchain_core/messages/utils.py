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
from collections.abc import Iterable, Sequence
from functools import partial
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
    overload,
)

from pydantic import Discriminator, Field, Tag

from langchain_core.exceptions import ErrorCode, create_message
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolCall, ToolMessage, ToolMessageChunk

if TYPE_CHECKING:
    from langchain_text_splitters import TextSplitter

    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompt_values import PromptValue
    from langchain_core.runnables.base import Runnable


def _get_type(v: Any) -> str:
    """Get the type associated with the object for serialization purposes."""
    if isinstance(v, dict) and "type" in v:
        return v["type"]
    elif hasattr(v, "type"):
        return v.type
    else:
        msg = (
            f"Expected either a dictionary with a 'type' key or an object "
            f"with a 'type' attribute. Instead got type {type(v)}."
        )
        raise TypeError(msg)


AnyMessage = Annotated[
    Union[
        Annotated[AIMessage, Tag(tag="ai")],
        Annotated[HumanMessage, Tag(tag="human")],
        Annotated[ChatMessage, Tag(tag="chat")],
        Annotated[SystemMessage, Tag(tag="system")],
        Annotated[FunctionMessage, Tag(tag="function")],
        Annotated[ToolMessage, Tag(tag="tool")],
        Annotated[AIMessageChunk, Tag(tag="AIMessageChunk")],
        Annotated[HumanMessageChunk, Tag(tag="HumanMessageChunk")],
        Annotated[ChatMessageChunk, Tag(tag="ChatMessageChunk")],
        Annotated[SystemMessageChunk, Tag(tag="SystemMessageChunk")],
        Annotated[FunctionMessageChunk, Tag(tag="FunctionMessageChunk")],
        Annotated[ToolMessageChunk, Tag(tag="ToolMessageChunk")],
    ],
    Field(discriminator=Discriminator(_get_type)),
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
            msg = f"Got unsupported message type: {m}"
            raise ValueError(msg)  # noqa: TRY004
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
        msg = f"Got unexpected message type: {_type}"
        raise ValueError(msg)


def messages_from_dict(messages: Sequence[dict]) -> list[BaseMessage]:
    """Convert a sequence of messages from dicts to Message objects.

    Args:
        messages: Sequence of messages (as dicts) to convert.

    Returns:
        list of messages (BaseMessages).
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
    BaseMessage, list[str], tuple[str, str], str, dict[str, Any]
]


def _create_message_from_message_type(
    message_type: str,
    content: str,
    name: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
    id: Optional[str] = None,
    **additional_kwargs: Any,
) -> BaseMessage:
    """Create a message from a message type and content string.

    Args:
        message_type: (str) the type of the message (e.g., "human", "ai", etc.).
        content: (str) the content string.
        name: (str) the name of the message. Default is None.
        tool_call_id: (str) the tool call id. Default is None.
        tool_calls: (list[dict[str, Any]]) the tool calls. Default is None.
        id: (str) the id of the message. Default is None.
        additional_kwargs: (dict[str, Any]) additional keyword arguments.

    Returns:
        a message of the appropriate type.

    Raises:
        ValueError: if the message type is not one of "human", "user", "ai",
            "assistant", "function", "tool", "system", or "developer".
    """
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if tool_call_id is not None:
        kwargs["tool_call_id"] = tool_call_id
    if additional_kwargs:
        if response_metadata := additional_kwargs.pop("response_metadata", None):
            kwargs["response_metadata"] = response_metadata
        kwargs["additional_kwargs"] = additional_kwargs  # type: ignore[assignment]
        additional_kwargs.update(additional_kwargs.pop("additional_kwargs", {}))
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
        if example := kwargs.get("additional_kwargs", {}).pop("example", False):
            kwargs["example"] = example
        message: BaseMessage = HumanMessage(content=content, **kwargs)
    elif message_type in ("ai", "assistant"):
        if example := kwargs.get("additional_kwargs", {}).pop("example", False):
            kwargs["example"] = example
        message = AIMessage(content=content, **kwargs)
    elif message_type in ("system", "developer"):
        if message_type == "developer":
            kwargs["additional_kwargs"] = kwargs.get("additional_kwargs") or {}
            kwargs["additional_kwargs"]["__openai_role__"] = "developer"
        message = SystemMessage(content=content, **kwargs)
    elif message_type == "function":
        message = FunctionMessage(content=content, **kwargs)
    elif message_type == "tool":
        artifact = kwargs.get("additional_kwargs", {}).pop("artifact", None)
        message = ToolMessage(content=content, artifact=artifact, **kwargs)
    elif message_type == "remove":
        message = RemoveMessage(**kwargs)
    else:
        msg = (
            f"Unexpected message type: '{message_type}'. Use one of 'human',"
            f" 'user', 'ai', 'assistant', 'function', 'tool', 'system', or 'developer'."
        )
        msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
        raise ValueError(msg)
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
        message: a representation of a message in one of the supported formats.

    Returns:
        an instance of a message or a message template.

    Raises:
        NotImplementedError: if the message type is not supported.
        ValueError: if the message dict does not contain the required keys.
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
            # None msg content is not allowed
            msg_content = msg_kwargs.pop("content") or ""
        except KeyError as e:
            msg = f"Message dict must contain 'role' and 'content' keys, got {message}"
            msg = create_message(
                message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE
            )
            raise ValueError(msg) from e
        _message = _create_message_from_message_type(
            msg_type, msg_content, **msg_kwargs
        )
    else:
        msg = f"Unsupported message type: {type(message)}"
        msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
        raise NotImplementedError(msg)

    return _message


def convert_to_messages(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
) -> list[BaseMessage]:
    """Convert a sequence of messages to a list of messages.

    Args:
        messages: Sequence of messages to convert.

    Returns:
        list of messages (BaseMessages).
    """
    # Import here to avoid circular imports
    from langchain_core.prompt_values import PromptValue

    if isinstance(messages, PromptValue):
        return messages.to_messages()
    return [_convert_to_message(m) for m in messages]


def _runnable_support(func: Callable) -> Callable:
    @overload
    def wrapped(
        messages: Literal[None] = None, **kwargs: Any
    ) -> Runnable[Sequence[MessageLikeRepresentation], list[BaseMessage]]: ...

    @overload
    def wrapped(
        messages: Sequence[MessageLikeRepresentation], **kwargs: Any
    ) -> list[BaseMessage]: ...

    def wrapped(
        messages: Union[Sequence[MessageLikeRepresentation], None] = None,
        **kwargs: Any,
    ) -> Union[
        list[BaseMessage],
        Runnable[Sequence[MessageLikeRepresentation], list[BaseMessage]],
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
    include_types: Optional[Sequence[Union[str, type[BaseMessage]]]] = None,
    exclude_types: Optional[Sequence[Union[str, type[BaseMessage]]]] = None,
    include_ids: Optional[Sequence[str]] = None,
    exclude_ids: Optional[Sequence[str]] = None,
) -> list[BaseMessage]:
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
    filtered: list[BaseMessage] = []
    for msg in messages:
        if (
            (exclude_names and msg.name in exclude_names)
            or (exclude_types and _is_message_type(msg, exclude_types))
            or (exclude_ids and msg.id in exclude_ids)
        ):
            continue
        else:
            pass

        # default to inclusion when no inclusion criteria given.
        if (
            not (include_types or include_ids or include_names)
            or (include_names and msg.name in include_names)
            or (include_types and _is_message_type(msg, include_types))
            or (include_ids and msg.id in include_ids)
        ):
            filtered.append(msg)
        else:
            pass

    return filtered


@_runnable_support
def merge_message_runs(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
    *,
    chunk_separator: str = "\n",
) -> list[BaseMessage]:
    """Merge consecutive Messages of the same type.

    **NOTE**: ToolMessages are not merged, as each has a distinct tool call id that
    can't be merged.

    Args:
        messages: Sequence Message-like objects to merge.
        chunk_separator: Specify the string to be inserted between message chunks.
        Default is "\n".

    Returns:
        list of BaseMessages with consecutive runs of message types merged into single
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
    merged: list[BaseMessage] = []
    for msg in messages:
        last = merged.pop() if merged else None
        if not last:
            merged.append(msg)
        elif isinstance(msg, ToolMessage) or not isinstance(msg, last.__class__):
            merged.extend([last, msg])
        else:
            last_chunk = _msg_to_chunk(last)
            curr_chunk = _msg_to_chunk(msg)
            if curr_chunk.response_metadata:
                curr_chunk.response_metadata.clear()
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
        Callable[[list[BaseMessage]], int],
        Callable[[BaseMessage], int],
        BaseLanguageModel,
    ],
    strategy: Literal["first", "last"] = "last",
    allow_partial: bool = False,
    end_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
    start_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
    include_system: bool = False,
    text_splitter: Optional[Union[Callable[[str], list[str]], TextSplitter]] = None,
) -> list[BaseMessage]:
    r"""Trim messages to be below a token count.

    trim_messages can be used to reduce the size of a chat history to a specified token
    count or specified message count.

    In either case, if passing the trimmed chat history back into a chat model
    directly, the resulting chat history should usually satisfy the following
    properties:

    1. The resulting chat history should be valid. Most chat models expect that chat
       history starts with either (1) a `HumanMessage` or (2) a `SystemMessage` followed
       by a `HumanMessage`. To achieve this, set `start_on="human"`.
       In addition, generally a `ToolMessage` can only appear after an `AIMessage`
       that involved a tool call.
       Please see the following link for more information about messages:
       https://python.langchain.com/docs/concepts/#messages
    2. It includes recent messages and drops old messages in the chat history.
       To achieve this set the `strategy="last"`.
    3. Usually, the new chat history should include the `SystemMessage` if it
       was present in the original chat history since the `SystemMessage` includes
       special instructions to the chat model. The `SystemMessage` is almost always
       the first message in the history if present. To achieve this set the
       `include_system=True`.

    **Note** The examples below show how to configure `trim_messages` to achieve
        a behavior consistent with the above properties.

    Args:
        messages: Sequence of Message-like objects to trim.
        max_tokens: Max token count of trimmed messages.
        token_counter: Function or llm for counting tokens in a BaseMessage or a list of
            BaseMessage. If a BaseLanguageModel is passed in then
            BaseLanguageModel.get_num_tokens_from_messages() will be used.
            Set to `len` to count the number of **messages** in the chat history.
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
        list of trimmed BaseMessages.

    Raises:
        ValueError: if two incompatible arguments are specified or an unrecognized
            ``strategy`` is specified.

    Example:
        Trim chat history based on token count, keeping the SystemMessage if
        present, and ensuring that the chat history starts with a HumanMessage (
        or a SystemMessage followed by a HumanMessage).

        .. code-block:: python

            from typing import list

            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                BaseMessage,
                SystemMessage,
                trim_messages,
            )

            messages = [
                SystemMessage("you're a good assistant, you always respond with a joke."),
                HumanMessage("i wonder why it's called langchain"),
                AIMessage(
                    'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
                ),
                HumanMessage("and who is harrison chasing anyways"),
                AIMessage(
                    "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
                ),
                HumanMessage("what do you call a speechless parrot"),
            ]


            trim_messages(
                messages,
                max_tokens=45,
                strategy="last",
                token_counter=ChatOpenAI(model="gpt-4o"),
                # Most chat models expect that chat history starts with either:
                # (1) a HumanMessage or
                # (2) a SystemMessage followed by a HumanMessage
                start_on="human",
                # Usually, we want to keep the SystemMessage
                # if it's present in the original history.
                # The SystemMessage has special instructions for the model.
                include_system=True,
                allow_partial=False,
            )

        .. code-block:: python

            [
                SystemMessage(content="you're a good assistant, you always respond with a joke."),
                HumanMessage(content='what do you call a speechless parrot'),
            ]

        Trim chat history based on the message count, keeping the SystemMessage if
        present, and ensuring that the chat history starts with a HumanMessage (
        or a SystemMessage followed by a HumanMessage).

            trim_messages(
                messages,
                # When `len` is passed in as the token counter function,
                # max_tokens will count the number of messages in the chat history.
                max_tokens=4,
                strategy="last",
                # Passing in `len` as a token counter function will
                # count the number of messages in the chat history.
                token_counter=len,
                # Most chat models expect that chat history starts with either:
                # (1) a HumanMessage or
                # (2) a SystemMessage followed by a HumanMessage
                start_on="human",
                # Usually, we want to keep the SystemMessage
                # if it's present in the original history.
                # The SystemMessage has special instructions for the model.
                include_system=True,
                allow_partial=False,
            )

        .. code-block:: python

            [
                SystemMessage(content="you're a good assistant, you always respond with a joke."),
                HumanMessage(content='and who is harrison chasing anyways'),
                AIMessage(content="Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"),
                HumanMessage(content='what do you call a speechless parrot'),
            ]


        Trim chat history using a custom token counter function that counts the
        number of tokens in each message.

        .. code-block:: python

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

            def dummy_token_counter(messages: list[BaseMessage]) -> int:
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
        msg = (
            f"'token_counter' expected to be a model that implements "
            f"'get_num_tokens_from_messages()' or a function. Received object of type "
            f"{type(token_counter)}."
        )
        raise ValueError(msg)

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
        msg = f"Unrecognized {strategy=}. Supported strategies are 'last' and 'first'."
        raise ValueError(msg)


def convert_to_openai_messages(
    messages: Union[MessageLikeRepresentation, Sequence[MessageLikeRepresentation]],
    *,
    text_format: Literal["string", "block"] = "string",
) -> Union[dict, list[dict]]:
    """Convert LangChain messages into OpenAI message dicts.

    Args:
        messages: Message-like object or iterable of objects whose contents are
            in OpenAI, Anthropic, Bedrock Converse, or VertexAI formats.
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
            - dict:
                If a single message-like object is passed in, a single OpenAI message
                dict is returned.
            - list[dict]:
                If a sequence of message-like objects are passed in, a list of OpenAI
                message dicts is returned.

    Example:

        .. code-block:: python

            from langchain_core.messages import (
                convert_to_openai_messages,
                AIMessage,
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
            oai_messages = convert_to_openai_messages(messages)
            # -> [
            #   {'role': 'system', 'content': 'foo'},
            #   {'role': 'user', 'content': [{'type': 'text', 'text': 'whats in this'}, {'type': 'image_url', 'image_url': {'url': "data:image/png;base64,'/9j/4AAQSk'"}}]},
            #   {'role': 'assistant', 'tool_calls': [{'type': 'function', 'id': '1','function': {'name': 'analyze', 'arguments': '{"baz": "buz"}'}}], 'content': ''},
            #   {'role': 'tool', 'name': 'bar', 'content': 'foobar'},
            #   {'role': 'assistant', 'content': 'thats nice'}
            # ]

    .. versionadded:: 0.3.11

    """  # noqa: E501
    if text_format not in ("string", "block"):
        err = f"Unrecognized {text_format=}, expected one of 'string' or 'block'."
        raise ValueError(err)

    oai_messages: list = []

    if is_single := isinstance(messages, (BaseMessage, dict, str)):
        messages = [messages]

    messages = convert_to_messages(messages)

    for i, message in enumerate(messages):
        oai_msg: dict = {"role": _get_message_openai_role(message)}
        tool_messages: list = []
        content: Union[str, list[dict]]

        if message.name:
            oai_msg["name"] = message.name
        if isinstance(message, AIMessage) and message.tool_calls:
            oai_msg["tool_calls"] = _convert_to_openai_tool_calls(message.tool_calls)
        if message.additional_kwargs.get("refusal"):
            oai_msg["refusal"] = message.additional_kwargs["refusal"]
        if isinstance(message, ToolMessage):
            oai_msg["tool_call_id"] = message.tool_call_id

        if not message.content:
            content = "" if text_format == "string" else []
        elif isinstance(message.content, str):
            if text_format == "string":
                content = message.content
            else:
                content = [{"type": "text", "text": message.content}]
        else:
            if text_format == "string" and all(
                isinstance(block, str) or block.get("type") == "text"
                for block in message.content
            ):
                content = "\n".join(
                    block if isinstance(block, str) else block["text"]
                    for block in message.content
                )
            else:
                content = []
                for j, block in enumerate(message.content):
                    # OpenAI format
                    if isinstance(block, str):
                        content.append({"type": "text", "text": block})
                    elif block.get("type") == "text":
                        if missing := [k for k in ("text",) if k not in block]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'text' "
                                f"but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        content.append({"type": block["type"], "text": block["text"]})
                    elif block.get("type") == "image_url":
                        if missing := [k for k in ("image_url",) if k not in block]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'image_url' "
                                f"but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": block["image_url"],
                            }
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
                                err = (
                                    f"Unrecognized content block at "
                                    f"messages[{i}].content[{j}] has 'type': 'image' "
                                    f"but 'source' is missing expected key(s) "
                                    f"{missing}. Full content block:\n\n{block}"
                                )
                                raise ValueError(err)
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
                                err = (
                                    f"Unrecognized content block at "
                                    f"messages[{i}].content[{j}] has key 'image', "
                                    f"but 'image' is missing expected key(s) "
                                    f"{missing}. Full content block:\n\n{block}"
                                )
                                raise ValueError(err)
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
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'image' "
                                f"but does not have a 'source' or 'image' key. Full "
                                f"content block:\n\n{block}"
                            )
                            raise ValueError(err)
                    elif block.get("type") == "tool_use":
                        if missing := [
                            k for k in ("id", "name", "input") if k not in block
                        ]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'tool_use', but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        if not any(
                            tool_call["id"] == block["id"]
                            for tool_call in cast(AIMessage, message).tool_calls
                        ):
                            oai_msg["tool_calls"] = oai_msg.get("tool_calls", [])
                            oai_msg["tool_calls"].append(
                                {
                                    "type": "function",
                                    "id": block["id"],
                                    "function": {
                                        "name": block["name"],
                                        "arguments": json.dumps(block["input"]),
                                    },
                                }
                            )
                    elif block.get("type") == "tool_result":
                        if missing := [
                            k for k in ("content", "tool_use_id") if k not in block
                        ]:
                            msg = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'tool_result', but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(msg)
                        tool_message = ToolMessage(
                            block["content"],
                            tool_call_id=block["tool_use_id"],
                            status="error" if block.get("is_error") else "success",
                        )
                        # Recurse to make sure tool message contents are OpenAI format.
                        tool_messages.extend(
                            convert_to_openai_messages(
                                [tool_message], text_format=text_format
                            )
                        )
                    elif (block.get("type") == "json") or "json" in block:
                        if "json" not in block:
                            msg = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'json' "
                                f"but does not have a 'json' key. Full "
                                f"content block:\n\n{block}"
                            )
                            raise ValueError(msg)
                        content.append(
                            {
                                "type": "text",
                                "text": json.dumps(block["json"]),
                            }
                        )
                    elif (
                        block.get("type") == "guard_content"
                    ) or "guard_content" in block:
                        if (
                            "guard_content" not in block
                            or "text" not in block["guard_content"]
                        ):
                            msg = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'guard_content' but does not have a "
                                f"messages[{i}].content[{j}]['guard_content']['text'] "
                                f"key. Full content block:\n\n{block}"
                            )
                            raise ValueError(msg)
                        text = block["guard_content"]["text"]
                        if isinstance(text, dict):
                            text = text["text"]
                        content.append({"type": "text", "text": text})
                    # VertexAI format
                    elif block.get("type") == "media":
                        if missing := [
                            k for k in ("mime_type", "data") if k not in block
                        ]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'media' but does not have key(s) {missing}. Full "
                                f"content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        if "image" not in block["mime_type"]:
                            err = (
                                f"OpenAI messages can only support text and image data."
                                f" Received content block with media of type:"
                                f" {block['mime_type']}"
                            )
                            raise ValueError(err)
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
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] does not match OpenAI, "
                            f"Anthropic, Bedrock Converse, or VertexAI format. Full "
                            f"content block:\n\n{block}"
                        )
                        raise ValueError(err)
                if text_format == "string" and not any(
                    block["type"] != "text" for block in content
                ):
                    content = "\n".join(block["text"] for block in content)
        oai_msg["content"] = content
        if message.content and not oai_msg["content"] and tool_messages:
            oai_messages.extend(tool_messages)
        else:
            oai_messages.extend([oai_msg, *tool_messages])

    if is_single:
        return oai_messages[0]
    else:
        return oai_messages


def _first_max_tokens(
    messages: Sequence[BaseMessage],
    *,
    max_tokens: int,
    token_counter: Callable[[list[BaseMessage]], int],
    text_splitter: Callable[[str], list[str]],
    partial_strategy: Optional[Literal["first", "last"]] = None,
    end_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
) -> list[BaseMessage]:
    messages = list(messages)
    if not messages:
        return messages
    idx = 0
    for i in range(len(messages)):
        if token_counter(messages[:-i] if i else messages) <= max_tokens:
            idx = len(messages) - i
            break
    if partial_strategy and (idx < len(messages) - 1 or idx == 0):
        included_partial = False
        if isinstance(messages[idx].content, list):
            excluded = messages[idx].model_copy(deep=True)
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
            excluded = messages[idx].model_copy(deep=True)
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
    token_counter: Callable[[list[BaseMessage]], int],
    text_splitter: Callable[[str], list[str]],
    allow_partial: bool = False,
    include_system: bool = False,
    start_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
    end_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
) -> list[BaseMessage]:
    messages = list(messages)
    if len(messages) == 0:
        return []
    if end_on:
        while messages and not _is_message_type(messages[-1], end_on):
            messages.pop()
    swapped_system = include_system and isinstance(messages[0], SystemMessage)
    reversed_ = messages[:1] + messages[1:][::-1] if swapped_system else messages[::-1]

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


_MSG_CHUNK_MAP: dict[type[BaseMessage], type[BaseMessageChunk]] = {
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
        return _MSG_CHUNK_MAP[message.__class__](**message.model_dump(exclude={"type"}))

    for msg_cls, chunk_cls in _MSG_CHUNK_MAP.items():
        if isinstance(message, msg_cls):
            return chunk_cls(**message.model_dump(exclude={"type"}))

    msg = (
        f"Unrecognized message class {message.__class__}. Supported classes are "
        f"{list(_MSG_CHUNK_MAP.keys())}"
    )
    msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
    raise ValueError(msg)


def _chunk_to_msg(chunk: BaseMessageChunk) -> BaseMessage:
    if chunk.__class__ in _CHUNK_MSG_MAP:
        return _CHUNK_MSG_MAP[chunk.__class__](
            **chunk.model_dump(exclude={"type", "tool_call_chunks"})
        )
    for chunk_cls, msg_cls in _CHUNK_MSG_MAP.items():
        if isinstance(chunk, chunk_cls):
            return msg_cls(**chunk.model_dump(exclude={"type", "tool_call_chunks"}))

    msg = (
        f"Unrecognized message chunk class {chunk.__class__}. Supported classes are "
        f"{list(_CHUNK_MSG_MAP.keys())}"
    )
    msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
    raise ValueError(msg)


def _default_text_splitter(text: str) -> list[str]:
    splits = text.split("\n")
    return [s + "\n" for s in splits[:-1]] + splits[-1:]


def _is_message_type(
    message: BaseMessage,
    type_: Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]],
) -> bool:
    types = [type_] if isinstance(type_, (str, type)) else type_
    types_str = [t for t in types if isinstance(t, str)]
    types_types = tuple(t for t in types if isinstance(t, type))

    return message.type in types_str or isinstance(message, types_types)


def _bytes_to_b64_str(bytes_: bytes) -> str:
    return base64.b64encode(bytes_).decode("utf-8")


def _get_message_openai_role(message: BaseMessage) -> str:
    if isinstance(message, AIMessage):
        return "assistant"
    elif isinstance(message, HumanMessage):
        return "user"
    elif isinstance(message, ToolMessage):
        return "tool"
    elif isinstance(message, SystemMessage):
        return message.additional_kwargs.get("__openai_role__", "system")
    elif isinstance(message, FunctionMessage):
        return "function"
    elif isinstance(message, ChatMessage):
        return message.role
    else:
        msg = f"Unknown BaseMessage type {message.__class__}."
        raise ValueError(msg)  # noqa: TRY004


def _convert_to_openai_tool_calls(tool_calls: list[ToolCall]) -> list[dict]:
    return [
        {
            "type": "function",
            "id": tool_call["id"],
            "function": {
                "name": tool_call["name"],
                "arguments": json.dumps(tool_call["args"]),
            },
        }
        for tool_call in tool_calls
    ]
