from typing import Any, List, Literal

from langchain_core._api import beta
from langchain_core.messages.base import BaseMessage


@beta()
class RemoveMessage(BaseMessage):
    """Message responsible for deleting other messages."""

    type: Literal["remove"] = "remove"

    def __init__(self, id: str, **kwargs: Any) -> None:
        if kwargs.pop("content", None):
            raise ValueError("RemoveMessage does not support 'content' field.")

        return super().__init__("", id=id, **kwargs)

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


RemoveMessage.update_forward_refs()
