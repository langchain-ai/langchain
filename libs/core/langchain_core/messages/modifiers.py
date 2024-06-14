from typing import Any, List, Literal

from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class MessageModifier(BaseMessage):
    """Message responsible for modifying other messages (deleting / updating.)"""

    def __init__(self, id: str) -> None:
        return super().__init__("modifier", id=id)


class RemoveMessageModifier(MessageModifier):
    """Message responsible for deleting other messages."""

    type: Literal["remove"] = "remove"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


RemoveMessageModifier.update_forward_refs()