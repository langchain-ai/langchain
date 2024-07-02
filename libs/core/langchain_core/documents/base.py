from __future__ import annotations

from typing import Any, List, Literal, Optional

from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Field


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

    # The ID field is optional at the moment.
    # It will likely become required in a future major release after
    # it has been adopted by enough vectorstore implementations.
    id: Optional[str] = None
    """An optional identifier for the document.
    
    Ideally this should be unique across the document collection and formatted 
    as a UUID, but this will not be enforced.
    """

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

    def __str__(self) -> str:
        """Override __str__ to restrict it to page_content and metadata."""
        # The format matches pydantic format for __str__.
        #
        # The purpose of this change is to make sure that user code that
        # feeds Document objects directly into prompts remains unchanged
        # due to the addition of the id field (or any other fields in the future).
        #
        # This override will likely be removed in the future in favor of
        # a more general solution of formatting content directly inside the prompts.
        if self.metadata:
            return f"page_content='{self.page_content}' metadata={self.metadata}"
        else:
            return f"page_content='{self.page_content}'"
