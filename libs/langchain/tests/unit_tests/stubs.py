from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import BaseModel


class _AnyIDMixin(BaseModel):
    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseModel):
            dump = self.model_dump()
            dump.pop("id")
            other_dump = other.model_dump()
            other_dump.pop("id")
            return dump == other_dump
        return False

    __hash__ = None  # type: ignore[assignment]


class _AnyIdAIMessage(AIMessage, _AnyIDMixin):
    """AIMessage with any ID."""


class _AnyIdAIMessageChunk(AIMessageChunk, _AnyIDMixin):
    """AIMessageChunk with any ID."""
