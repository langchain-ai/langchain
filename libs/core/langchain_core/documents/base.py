from __future__ import annotations

from typing import Any, List, Literal

from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Field


class Document(Serializable):
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""
    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """
    type: Literal["Document"] = "Document"

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Pass page_content in as positional or named arg."""
        super().__init__(page_content=page_content, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "document"]


class DocumentSearchHit(Document):
    """Class for storing a document and fields associated with retrieval."""

    score: float
    """Score associated with the document's relevance to a query."""
    type: Literal["DocumentSearchHit"] = "DocumentSearchHit"  # type: ignore[assignment] # noqa: E501

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "document_search_hit"]
