from typing import List, Literal

from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class HumanMessage(BaseMessage):
    """A Message from a human."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    type: Literal["human"] = "human"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


HumanMessage.update_forward_refs()


class HumanMessageChunk(HumanMessage, BaseMessageChunk):
    """A Human Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["HumanMessageChunk"] = "HumanMessageChunk"  # type: ignore[assignment] # noqa: E501

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]
