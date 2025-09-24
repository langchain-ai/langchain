"""Human message."""

from typing import Any, Literal, Union

from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class HumanMessage(BaseMessage):
    """Message from a human.

    ``HumanMessage``s are messages that are passed in from a human to the model.

    Example:

        .. code-block:: python

            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content="You are a helpful assistant! Your name is Bob."),
                HumanMessage(content="What is your name?"),
            ]

            # Instantiate a chat model and invoke it with the messages
            model = ...
            print(model.invoke(messages))

    """

    example: bool = False
    """Use to denote that a message is part of an example conversation.

    At the moment, this is ignored by most models. Usage is discouraged.
    Defaults to False.

    """

    type: Literal["human"] = "human"
    """The type of the message (used for serialization).

    Defaults to ``'human'``.

    """

    def __init__(
        self,
        content: Union[str, list[Union[str, dict]]],
        **kwargs: Any,
    ) -> None:
        """Initialize ``HumanMessage``.

        Args:
            content: The string contents of the message.
            kwargs: Additional fields to pass to the message.
        """
        super().__init__(content=content, **kwargs)


class HumanMessageChunk(HumanMessage, BaseMessageChunk):
    """Human Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["HumanMessageChunk"] = "HumanMessageChunk"  # type: ignore[assignment]
    """The type of the message (used for serialization).
    Defaults to "HumanMessageChunk"."""
