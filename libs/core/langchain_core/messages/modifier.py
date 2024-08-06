from typing import Any, List, Literal

from langchain_core._api import beta
from langchain_core.messages.base import BaseMessage


@beta()
class RemoveMessage(BaseMessage):
    """Message responsible for deleting other messages."""

    type: Literal["remove"] = "remove"
    """The type of the message (used for serialization). Defaults to "remove"."""

    def __init__(self, id: str, **kwargs: Any) -> None:
        """Create a RemoveMessage.

        Args:
            id: The ID of the message to remove.
            kwargs: Additional fields to pass to the message.

        Raises:
            ValueError: If the 'content' field is passed in kwargs.
        """
        if kwargs.pop("content", None):
            raise ValueError("RemoveMessage does not support 'content' field.")

        return super().__init__("", id=id, **kwargs)

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object.
        Default is ["langchain", "schema", "messages"]."""
        return ["langchain", "schema", "messages"]


RemoveMessage.model_rebuild()
