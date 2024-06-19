from typing import Any, Dict, List, Literal, Union

from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class SystemMessage(BaseMessage):
    """Message for priming AI behavior, usually passed in as the first of a sequence
    of input messages.
    """

    type: Literal["system"] = "system"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    def __init__(
        self, content: Union[str, List[Union[str, Dict]]], **kwargs: Any
    ) -> None:
        """Pass in content as positional arg."""
        super().__init__(content=content, **kwargs)


SystemMessage.update_forward_refs()


class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """System Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["SystemMessageChunk"] = "SystemMessageChunk"  # type: ignore[assignment]

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]
