"""Base message."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union, cast, overload

from pydantic import ConfigDict, Field
from typing_extensions import Self

from langchain_core._api.deprecation import warn_deprecated
from langchain_core.load.serializable import Serializable
from langchain_core.messages import content as types
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.prompts.chat import ChatPromptTemplate


class TextAccessor(str):
    """String-like object that supports both property and method access patterns.

    Exists to maintain backward compatibility while transitioning from method-based to
    property-based text access in message objects. In LangChain <v1.0, message text was
    accessed via ``.text()`` method calls. In v1.0=<, the preferred pattern is property
    access via ``.text``.

    Rather than breaking existing code immediately, ``TextAccessor`` allows both
    patterns:
    - Modern property access: ``message.text`` (returns string directly)
    - Legacy method access: ``message.text()`` (callable, emits deprecation warning)

    """

    __slots__ = ()

    def __new__(cls, value: str) -> Self:
        """Create new TextAccessor instance."""
        return str.__new__(cls, value)

    def __call__(self) -> str:
        """Enable method-style text access for backward compatibility.

        This method exists solely to support legacy code that calls ``.text()``
        as a method. New code should use property access (``.text``) instead.

        .. deprecated:: 1.0.0
            Calling ``.text()`` as a method is deprecated. Use ``.text`` as a property
            instead. This method will be removed in 2.0.0.

        Returns:
            The string content, identical to property access.

        """
        warn_deprecated(
            since="1.0.0",
            message=(
                "Calling .text() as a method is deprecated. "
                "Use .text as a property instead (e.g., message.text)."
            ),
            removal="2.0.0",
        )
        return str(self)


class BaseMessage(Serializable):
    """Base abstract message class.

    Messages are the inputs and outputs of ChatModels.
    """

    content: Union[str, list[Union[str, dict]]]
    """The string contents of the message."""

    additional_kwargs: dict = Field(default_factory=dict)
    """Reserved for additional payload data associated with the message.

    For example, for a message from an AI, this could include tool calls as
    encoded by the model provider.
    """

    response_metadata: dict = Field(default_factory=dict)
    """Response metadata. For example: response headers, logprobs, token counts, model
    name."""

    type: str
    """The type of the message. Must be a string that is unique to the message type.

    The purpose of this field is to allow for easy identification of the message type
    when deserializing messages.
    """

    name: Optional[str] = None
    """An optional name for the message.

    This can be used to provide a human-readable name for the message.

    Usage of this field is optional, and whether it's used or not is up to the
    model implementation.
    """

    id: Optional[str] = Field(default=None, coerce_numbers_to_str=True)
    """An optional unique identifier for the message. This should ideally be
    provided by the provider/model which created the message."""

    model_config = ConfigDict(
        extra="allow",
    )

    @overload
    def __init__(
        self,
        content: Union[str, list[Union[str, dict]]],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        content: Optional[Union[str, list[Union[str, dict]]]] = None,
        content_blocks: Optional[list[types.ContentBlock]] = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        content: Optional[Union[str, list[Union[str, dict]]]] = None,
        content_blocks: Optional[list[types.ContentBlock]] = None,
        **kwargs: Any,
    ) -> None:
        """Specify ``content`` as positional arg or ``content_blocks`` for typing."""
        if content_blocks is not None:
            super().__init__(content=content_blocks, **kwargs)
        else:
            super().__init__(content=content, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """BaseMessage is serializable.

        Returns:
            True
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        Default is ["langchain", "schema", "messages"].
        """
        return ["langchain", "schema", "messages"]

    @property
    def content_blocks(self) -> list[types.ContentBlock]:
        r"""Return ``content`` as a list of standardized :class:`~langchain_core.messages.content.ContentBlock`\s.

        .. important::

            To use this property correctly, the corresponding ``ChatModel`` must support
            ``message_version='v1'`` or higher (and it must be set):

            .. code-block:: python

                from langchain.chat_models import init_chat_model
                llm = init_chat_model("...", message_version="v1")

                # or

                from langchain-openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o", message_version="v1")

            Otherwise, the property will perform best-effort parsing to standard types,
            though some content may be misinterpreted.

        .. versionadded:: 1.0.0

        """  # noqa: E501
        from langchain_core.messages import content as types
        from langchain_core.messages.block_translators.anthropic import (
            _convert_to_v1_from_anthropic_input,
        )
        from langchain_core.messages.block_translators.langchain_v0 import (
            _convert_v0_multimodal_input_to_v1,
        )
        from langchain_core.messages.block_translators.openai import (
            _convert_to_v1_from_chat_completions_input,
        )

        blocks: list[types.ContentBlock] = []

        # First pass: convert to standard blocks
        content = (
            [self.content]
            if isinstance(self.content, str) and self.content
            else self.content
        )
        for item in content:
            if isinstance(item, str):
                blocks.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type not in types.KNOWN_BLOCK_TYPES:
                    blocks.append({"type": "non_standard", "value": item})
                else:
                    blocks.append(cast("types.ContentBlock", item))

        # Subsequent passes: attempt to unpack non-standard blocks
        for parsing_step in [
            _convert_v0_multimodal_input_to_v1,
            _convert_to_v1_from_chat_completions_input,
            _convert_to_v1_from_anthropic_input,
        ]:
            blocks = parsing_step(blocks)
        return blocks

    @property
    def text(self) -> TextAccessor:
        """Get the text content of the message as a string.

        Can be used as both property (``message.text``) and method (``message.text()``).

        .. deprecated:: 1.0.0
            Calling ``.text()`` as a method is deprecated. Use ``.text`` as a property
            instead. This method will be removed in 2.0.0.

        Returns:
            The text content of the message.
        """
        if isinstance(self.content, str):
            text_value = self.content
        else:
            # must be a list
            blocks = [
                block
                for block in self.content
                if isinstance(block, str)
                or (block.get("type") == "text" and isinstance(block.get("text"), str))
            ]
            text_value = "".join(
                block if isinstance(block, str) else block["text"] for block in blocks
            )
        return TextAccessor(text_value)

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """Concatenate this message with another message."""
        from langchain_core.prompts.chat import ChatPromptTemplate

        prompt = ChatPromptTemplate(messages=[self])
        return prompt + other

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """Get a pretty representation of the message.

        Args:
            html: Whether to format the message as HTML. If True, the message will be
                formatted with HTML tags. Default is False.

        Returns:
            A pretty representation of the message.
        """
        title = get_msg_title_repr(self.type.title() + " Message", bold=html)
        # TODO: handle non-string content.
        if self.name is not None:
            title += f"\nName: {self.name}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        """Print a pretty representation of the message."""
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


def merge_content(
    first_content: Union[str, list[Union[str, dict]]],
    *contents: Union[str, list[Union[str, dict]]],
) -> Union[str, list[Union[str, dict]]]:
    """Merge multiple message contents.

    Args:
        first_content: The first content. Can be a string or a list.
        contents: The other contents. Can be a string or a list.

    Returns:
        The merged content.
    """
    merged: Union[str, list[Union[str, dict]]]
    merged = "" if first_content is None else first_content

    for content in contents:
        # If current is a string
        if isinstance(merged, str):
            # If the next chunk is also a string, then merge them naively
            if isinstance(content, str):
                merged += content
            # If the next chunk is a list, add the current to the start of the list
            else:
                merged = [merged, *content]
        elif isinstance(content, list):
            # If both are lists
            merged = merge_lists(cast("list", merged), content)  # type: ignore[assignment]
        # If the first content is a list, and the second content is a string
        # If the last element of the first content is a string
        # Add the second content to the last element
        elif merged and isinstance(merged[-1], str):
            merged[-1] += content
        # If second content is an empty string, treat as a no-op
        elif content == "":
            pass
        # Otherwise, add the second content as a new element of the list
        elif merged:
            merged.append(content)
    return merged


class BaseMessageChunk(BaseMessage):
    """Message chunk, which can be concatenated with other Message chunks."""

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        """Message chunks support concatenation with other message chunks.

        This functionality is useful to combine message chunks yielded from
        a streaming model into a complete message.

        Args:
            other: Another message chunk to concatenate with this one.

        Returns:
            A new message chunk that is the concatenation of this message chunk
            and the other message chunk.

        Raises:
            TypeError: If the other object is not a message chunk.

        For example,

        `AIMessageChunk(content="Hello") + AIMessageChunk(content=" World")`

        will give `AIMessageChunk(content="Hello World")`
        """
        if isinstance(other, BaseMessageChunk):
            # If both are (subclasses of) BaseMessageChunk,
            # concat into a single BaseMessageChunk

            return self.__class__(
                id=self.id,
                type=self.type,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
            )
        if isinstance(other, list) and all(
            isinstance(o, BaseMessageChunk) for o in other
        ):
            content = merge_content(self.content, *(o.content for o in other))
            additional_kwargs = merge_dicts(
                self.additional_kwargs, *(o.additional_kwargs for o in other)
            )
            response_metadata = merge_dicts(
                self.response_metadata, *(o.response_metadata for o in other)
            )
            return self.__class__(  # type: ignore[call-arg]
                id=self.id,
                content=content,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
            )
        msg = (
            'unsupported operand type(s) for +: "'
            f"{self.__class__.__name__}"
            f'" and "{other.__class__.__name__}"'
        )
        raise TypeError(msg)


def message_to_dict(message: BaseMessage) -> dict:
    """Convert a Message to a dictionary.

    Args:
        message: Message to convert.

    Returns:
        Message as a dict. The dict will have a "type" key with the message type
        and a "data" key with the message data as a dict.
    """
    return {"type": message.type, "data": message.model_dump()}


def messages_to_dict(messages: Sequence[BaseMessage]) -> list[dict]:
    """Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as BaseMessages) to convert.

    Returns:
        List of messages as dicts.
    """
    return [message_to_dict(m) for m in messages]


def get_msg_title_repr(title: str, *, bold: bool = False) -> str:
    """Get a title representation for a message.

    Args:
        title: The title.
        bold: Whether to bold the title. Default is False.

    Returns:
        The title representation.
    """
    padded = " " + title + " "
    sep_len = (80 - len(padded)) // 2
    sep = "=" * sep_len
    second_sep = sep + "=" if len(padded) % 2 else sep
    if bold:
        padded = get_bolded_text(padded)
    return f"{sep}{padded}{second_sep}"
