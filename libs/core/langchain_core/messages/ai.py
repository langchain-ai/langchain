from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.pydantic_v1 import root_validator


class AIMessage(BaseMessage):
    """Message from an AI."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    type: Literal["ai"] = "ai"

    name: Optional[str] = None

    @root_validator
    def sync_name(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Sync the name of the message with the name of the function."""
        additional_kwargs = values.get("additional_kwargs", {})
        if "name" in values and values["name"] is not None:
            if (
                "name" in additional_kwargs
                and values["name"] != additional_kwargs["name"]
            ):
                raise ValueError(
                    "Name is defined differently in name and the "
                    'additional_kwargs["name"].'
                )
            additional_kwargs["name"] = values["name"]
            values["additional_kwargs"] = additional_kwargs
        elif "name" in additional_kwargs:
            values["name"] = additional_kwargs["name"]
        return values

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


AIMessage.update_forward_refs()


class AIMessageChunk(AIMessage, BaseMessageChunk):
    """Message chunk from an AI."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["AIMessageChunk"] = "AIMessageChunk"  # type: ignore[assignment] # noqa: E501

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, AIMessageChunk):
            if self.example != other.example:
                raise ValueError(
                    "Cannot concatenate AIMessageChunks with different example values."
                )

            return self.__class__(
                example=self.example,
                content=merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)
