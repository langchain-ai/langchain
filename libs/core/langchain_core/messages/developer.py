from typing import Any, Literal, Union

from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class DeveloperMessage(BaseMessage):
    """Message from a developer.

    DeveloperMessages are messages that convey information from the developer
    to the system or other components within the architecture.

    Example:

        .. code-block:: python

            from langchain_core.messages import DeveloperMessage, SystemMessage

            messages = [
                SystemMessage(
                    content="You are a helpful assistant! Your name is Bob."
                ),
                DeveloperMessage(
                    content="Update the system configuration to version 2.0."
                )
            ]

    """
    type: Literal["developer"] = "developer"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.
        Default is ["langchain", "schema", "messages"]."""
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


class DeveloperMessageChunk(DeveloperMessage, BaseMessageChunk):
    """Developer Message chunk.

    This class handles segments of developer messages, allowing for the
    processing or transmission of these messages in parts.

    """
    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["DeveloperMessageChunk"] = "DeveloperMessageChunk"  # type: ignore[assignment]
    """The type of the message (used for serialization).
    Defaults to "DeveloperMessageChunk"."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.
        Default is ["langchain", "schema", "messages"]."""
        return ["langchain", "schema", "messages"]
