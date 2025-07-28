"""Common utilities and types for Cohere integration."""

from typing import Any, Dict, Optional


class CohereCitation:
    """Represents a citation from Cohere model responses.
    
    To use this, you need to install the ``langchain-cohere`` package.
    
    .. code-block:: bash
    
        pip install langchain-cohere
    
    Example:
        .. code-block:: python
        
            from langchain_cohere import CohereCitation
            
            citation = CohereCitation(
                start=0,
                end=10,
                text="cited text",
                document_ids=["doc1"]
            )
    
    """
    
    start: int
    """Start index of the citation in the generated text."""
    
    end: int
    """End index of the citation in the generated text."""
    
    text: str
    """The cited text."""
    
    document_ids: list[str]
    """List of document IDs that support this citation."""
    
    def __init__(
        self,
        start: int,
        end: int,
        text: str,
        document_ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Initialize citation."""
        self.start = start
        self.end = end
        self.text = text
        self.document_ids = document_ids