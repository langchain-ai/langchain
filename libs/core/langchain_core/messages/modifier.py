from typing import Any, List, Literal

from langchain_core.messages.base import BaseMessage


class RemoveMessage(BaseMessage):
    """Message responsible for deleting other messages."""

    type: Literal["remove"] = "remove"

    def __init__(self, id: str, **kwargs: Any) -> None:
        return super().__init__("", id=id)

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


RemoveMessage.update_forward_refs()