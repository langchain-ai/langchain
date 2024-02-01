from __future__ import annotations

from typing import Any, List, Literal, Optional

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

    def __init__(
        self, page_content: str, metadata: Optional[dict] = None, **kwargs: Any
    ) -> None:
        metadata = metadata or {}
        super().__init__(page_content=page_content, metadata=metadata, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "document"]
