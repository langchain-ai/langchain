from typing import Any, List, Literal

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.utils._merge import merge_dicts


class FunctionMessage(BaseMessage):
    """Message for passing the result of executing a function back to a model."""

    name: str
    """The name of the function that was executed."""

    type: Literal["function"] = "function"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


FunctionMessage.update_forward_refs()


class FunctionMessageChunk(FunctionMessage, BaseMessageChunk):
    """Function Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["FunctionMessageChunk"] = "FunctionMessageChunk"  # type: ignore[assignment]

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, FunctionMessageChunk):
            if self.name != other.name:
                raise ValueError(
                    "Cannot concatenate FunctionMessageChunks with different names."
                )

            return self.__class__(
                name=self.name,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
            )

        return super().__add__(other)
