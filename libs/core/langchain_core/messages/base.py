"""Base message."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union, cast

from pydantic import ConfigDict, Field

from langchain_core.load.serializable import Serializable
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.prompts.chat import ChatPromptTemplate


class BaseMessage(Serializable):
    """Base abstract message class.

    Messages are the inputs and outputs of ``ChatModel``s.
    """

    content: Union[str, list[Union[str, dict]]]
    """The string contents of the message."""

    additional_kwargs: dict = Field(default_factory=dict)
    """Reserved for additional payload data associated with the message.

    For example, for a message from an AI, this could include tool calls as
    encoded by the model provider.

    """

    response_metadata: dict = Field(default_factory=dict)
    """Examples: response headers, logprobs, token counts, model name."""

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
    """An optional unique identifier for the message.

    This should ideally be provided by the provider/model which created the message.

    """

    model_config = ConfigDict(
        extra="allow",
    )

    def __init__(
        self,
        content: Union[str, list[Union[str, dict]]],
        **kwargs: Any,
    ) -> None:
        """Initialize ``BaseMessage``.

        Args:
            content: The string contents of the message.
        """
        super().__init__(content=content, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """``BaseMessage`` is serializable.

        Returns:
            True
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        Returns:
            ``["langchain", "schema", "messages"]``
        """
        return ["langchain", "schema", "messages"]

    def text(self) -> str:
        """Get the text ``content`` of the message.

        Returns:
            The text content of the message.

        """
        if isinstance(self.content, str):
            return self.content

        # must be a list
        blocks = [
            block
            for block in self.content
            if isinstance(block, str)
            or (block.get("type") == "text" and isinstance(block.get("text"), str))
        ]
        return "".join(
            block if isinstance(block, str) else block["text"] for block in blocks
        )

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """Concatenate this message with another message.

        Args:
            other: Another message to concatenate with this one.

        Returns:
            A ChatPromptTemplate containing both messages.
        """
        # Import locally to prevent circular imports.
        from langchain_core.prompts.chat import ChatPromptTemplate  # noqa: PLC0415

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
        first_content: The first ``content``. Can be a string or a list.
        contents: The other ``content``s. Can be a string or a list.

    Returns:
        The merged content.

    """
    merged = first_content
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
        elif content:
            # Otherwise, add the second content as a new element of the list
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

        ``AIMessageChunk(content="Hello") + AIMessageChunk(content=" World")``

        will give ``AIMessageChunk(content="Hello World")``

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
        Message as a dict. The dict will have a ``type`` key with the message type
        and a ``data`` key with the message data as a dict.

    """
    return {"type": message.type, "data": message.model_dump()}


def messages_to_dict(messages: Sequence[BaseMessage]) -> list[dict]:
    """Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as ``BaseMessage``s) to convert.

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
