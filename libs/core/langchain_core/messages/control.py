from typing import Any, Literal, Union

from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class ControlMessage(BaseMessage):
    """Message from a control role.

    ControlMessages are messages that are passed in by a user to control a toggleable
     model functionality.

    Example:

        .. code-block:: python

            from langchain_core.messages import ControlMessage, HumanMessage

            messages = [
                ControlMessage(
                    content="thinking"
                ),
                HumanMessage(
                    content="What is your name?"
                )
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

    type: Literal["control"] = "control"
    """The type of the message (used for serialization). Defaults to "control"."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.
        Default is ["langchain", "schema", "messages"].
        """
        return ["langchain", "schema", "messages"]

    def __init__(
        self, content: Union[str, list[Union[str, dict]]], **kwargs: Any
    ) -> None:
        """Pass in content as positional arg.

        Args:
            content: The string contents of the message.
            kwargs: Additional fields to pass to the message.
        """
        super().__init__(content=content, **kwargs)


ControlMessage.model_rebuild()


class ControlMessageChunk(ControlMessage, BaseMessageChunk):
    """Control Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ControlMessageChunk"] = "ControlMessageChunk"  # type: ignore[assignment]
    """The type of the message (used for serialization).
    Defaults to "ControlMessageChunk"."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.
        Default is ["langchain", "schema", "messages"].
        """
        return ["langchain", "schema", "messages"]
