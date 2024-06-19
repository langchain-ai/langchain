from __future__ import annotations

from dataclasses import field
from typing import Any, List, Literal

from langchain_core.load.serializable import Serializable


class Document(Serializable):
    """Class for storing a piece of text and associated metadata.

    Example:

        .. code-block:: python

            from langchain_core.documents import Document

            document = Document(
                page_content="Hello, world!",
                metadata={"source": "https://example.com"}
            )
    """

    page_content: str = field(kw_only=False)
    """String text."""
    metadata: dict = field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """
    type: Literal["Document"] = "Document"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "document"]
