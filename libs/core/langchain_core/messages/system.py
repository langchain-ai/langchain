"""System message."""

from typing import Any, Literal, Optional, Union, cast, overload

from langchain_core.messages import content as types
from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class SystemMessage(BaseMessage):
    """Message for priming AI behavior.

    The system message is usually passed in as the first of a sequence
    of input messages.

    Example:

        .. code-block:: python

            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(
                    content="You are a helpful assistant! Your name is Bob."
                ),
                HumanMessage(
                    content="What is your name?"
                )
            ]

            # Define a chat model and invoke it with the messages
            print(model.invoke(messages))

    """

    type: Literal["system"] = "system"
    """The type of the message (used for serialization). Defaults to "system"."""

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
            super().__init__(
                content=cast("Union[str, list[Union[str, dict]]]", content_blocks),
                **kwargs,
            )
        else:
            super().__init__(content=content, **kwargs)


class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """System Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["SystemMessageChunk"] = "SystemMessageChunk"  # type: ignore[assignment]
    """The type of the message (used for serialization).
    Defaults to "SystemMessageChunk"."""
