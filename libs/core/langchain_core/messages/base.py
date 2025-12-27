"""Base message."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload

from pydantic import ConfigDict, Field

from langchain_core._api.deprecation import warn_deprecated
from langchain_core.load.serializable import Serializable
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from langchain_core.messages import content as types
    from langchain_core.prompts.chat import ChatPromptTemplate


def _extract_reasoning_from_additional_kwargs(
    message: BaseMessage,
) -> types.ReasoningContentBlock | None:
    """Extract `reasoning_content` from `additional_kwargs`.

    Handles reasoning content stored in various formats:
    - `additional_kwargs["reasoning_content"]` (string) - Ollama, DeepSeek, XAI, Groq

    Args:
        message: The message to extract reasoning from.

    Returns:
        A `ReasoningContentBlock` if reasoning content is found, None otherwise.
    """
    additional_kwargs = getattr(message, "additional_kwargs", {})

    reasoning_content = additional_kwargs.get("reasoning_content")
    if reasoning_content is not None and isinstance(reasoning_content, str):
        return {"type": "reasoning", "reasoning": reasoning_content}

    return None


class TextAccessor(str):
    """String-like object that supports both property and method access patterns.

    Exists to maintain backward compatibility while transitioning from method-based to
    property-based text access in message objects. In LangChain <v1.0, message text was
    accessed via `.text()` method calls. In v1.0=<, the preferred pattern is property
    access via `.text`.

    Rather than breaking existing code immediately, `TextAccessor` allows both
    patterns:
    - Modern property access: `message.text` (returns string directly)
    - Legacy method access: `message.text()` (callable, emits deprecation warning)

    """

    __slots__ = ()

    def __new__(cls, value: str) -> Self:
        """Create new TextAccessor instance."""
        return str.__new__(cls, value)

    def __call__(self) -> str:
        """Enable method-style text access for backward compatibility.

        This method exists solely to support legacy code that calls `.text()`
        as a method. New code should use property access (`.text`) instead.

        !!! deprecated
            As of `langchain-core` 1.0.0, calling `.text()` as a method is deprecated.
            Use `.text` as a property instead. This method will be removed in 2.0.0.

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

    Messages are the inputs and outputs of a chat model.

    Examples include [`HumanMessage`][langchain.messages.HumanMessage],
    [`AIMessage`][langchain.messages.AIMessage], and
    [`SystemMessage`][langchain.messages.SystemMessage].
    """

    content: str | list[str | dict]
    """The contents of the message."""

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

    name: str | None = None
    """An optional name for the message.

    This can be used to provide a human-readable name for the message.

    Usage of this field is optional, and whether it's used or not is up to the
    model implementation.

    """

    id: str | None = Field(default=None, coerce_numbers_to_str=True)
    """An optional unique identifier for the message.

    This should ideally be provided by the provider/model which created the message.

    """

    model_config = ConfigDict(
        extra="allow",
    )

    @overload
    def __init__(
        self,
        content: str | list[str | dict],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        content: str | list[str | dict] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        content: str | list[str | dict] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `BaseMessage`.

        Specify `content` as positional arg or `content_blocks` for typing.

        Args:
            content: The contents of the message.
            content_blocks: Typed standard content.
            **kwargs: Additional arguments to pass to the parent class.
        """
        if content_blocks is not None:
            super().__init__(content=content_blocks, **kwargs)
        else:
            super().__init__(content=content, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """`BaseMessage` is serializable.

        Returns:
            True
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "messages"]`
        """
        return ["langchain", "schema", "messages"]

    @property
    def content_blocks(self) -> list[types.ContentBlock]:
        r"""Load content blocks from the message content.

        !!! version-added "Added in `langchain-core` 1.0.0"

        """
        # Needed here to avoid circular import, as these classes import BaseMessages
        from langchain_core.messages import content as types  # noqa: PLC0415
        from langchain_core.messages.block_translators.anthropic import (  # noqa: PLC0415
            _convert_to_v1_from_anthropic_input,
        )
        from langchain_core.messages.block_translators.bedrock_converse import (  # noqa: PLC0415
            _convert_to_v1_from_converse_input,
        )
        from langchain_core.messages.block_translators.google_genai import (  # noqa: PLC0415
            _convert_to_v1_from_genai_input,
        )
        from langchain_core.messages.block_translators.langchain_v0 import (  # noqa: PLC0415
            _convert_v0_multimodal_input_to_v1,
        )
        from langchain_core.messages.block_translators.openai import (  # noqa: PLC0415
            _convert_to_v1_from_chat_completions_input,
        )

        blocks: list[types.ContentBlock] = []
        content = (
            # Transpose string content to list, otherwise assumed to be list
            [self.content]
            if isinstance(self.content, str) and self.content
            else self.content
        )
        for item in content:
            if isinstance(item, str):
                # Plain string content is treated as a text block
                blocks.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type not in types.KNOWN_BLOCK_TYPES:
                    # Handle all provider-specific or None type blocks as non-standard -
                    # we'll come back to these later
                    blocks.append({"type": "non_standard", "value": item})
                else:
                    # Guard against v0 blocks that share the same `type` keys
                    if "source_type" in item:
                        blocks.append({"type": "non_standard", "value": item})
                        continue

                    # This can't be a v0 block (since they require `source_type`),
                    # so it's a known v1 block type
                    blocks.append(cast("types.ContentBlock", item))

        # Subsequent passes: attempt to unpack non-standard blocks.
        # This is the last stop - if we can't parse it here, it is left as non-standard
        for parsing_step in [
            _convert_v0_multimodal_input_to_v1,
            _convert_to_v1_from_chat_completions_input,
            _convert_to_v1_from_anthropic_input,
            _convert_to_v1_from_genai_input,
            _convert_to_v1_from_converse_input,
        ]:
            blocks = parsing_step(blocks)
        return blocks

    @property
    def text(self) -> TextAccessor:
        """Get the text content of the message as a string.

        Can be used as both property (`message.text`) and method (`message.text()`).

        !!! deprecated
            As of `langchain-core` 1.0.0, calling `.text()` as a method is deprecated.
            Use `.text` as a property instead. This method will be removed in 2.0.0.

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
        """Concatenate this message with another message.

        Args:
            other: Another message to concatenate with this one.

        Returns:
            A ChatPromptTemplate containing both messages.
        """
        # Import locally to prevent circular imports.
        from langchain_core.prompts.chat import ChatPromptTemplate  # noqa: PLC0415

        prompt = ChatPromptTemplate(messages=[self])
        return prompt.__add__(other)

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """Get a pretty representation of the message.

        Args:
            html: Whether to format the message as HTML. If `True`, the message will be
                formatted with HTML tags.

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
    first_content: str | list[str | dict],
    *contents: str | list[str | dict],
) -> str | list[str | dict]:
    """Merge multiple message contents.

    Args:
        first_content: The first `content`. Can be a string or a list.
        contents: The other `content`s. Can be a string or a list.

    Returns:
        The merged content.

    """
    merged: str | list[str | dict]
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

        Example:
            ```txt
              AIMessageChunk(content="Hello", ...)
            + AIMessageChunk(content=" World", ...)
            = AIMessageChunk(content="Hello World", ...)
            ```
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
        Message as a dict. The dict will have a `type` key with the message type
        and a `data` key with the message data as a dict.

    """
    return {"type": message.type, "data": message.model_dump()}


def messages_to_dict(messages: Sequence[BaseMessage]) -> list[dict]:
    """Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as `BaseMessage`s) to convert.

    Returns:
        List of messages as dicts.

    """
    return [message_to_dict(m) for m in messages]


def get_msg_title_repr(title: str, *, bold: bool = False) -> str:
    """Get a title representation for a message.

    Args:
        title: The title.
        bold: Whether to bold the title.

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
