from typing import List, Literal

from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class SystemMessage(BaseMessage):
    """A Message for priming AI behavior, usually passed in as the first of a sequence
    of input messages.
    """

    type: Literal["system"] = "system"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


SystemMessage.update_forward_refs()


class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """A System Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["SystemMessageChunk"] = "SystemMessageChunk"  # type: ignore[assignment] # noqa: E501

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]
