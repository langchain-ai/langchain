"""Cohere embeddings."""

from typing import List, Optional

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    # Fallback for when langchain_core is not available
    class Embeddings:
        pass


class CohereEmbeddings(Embeddings):
    """Cohere embeddings.
    
    To use this, you need to install the ``langchain-cohere`` package.
    
    .. code-block:: bash
    
        pip install langchain-cohere
    
    Example:
        .. code-block:: python
        
            from langchain_cohere import CohereEmbeddings
            
            embeddings = CohereEmbeddings(model="embed-english-v3.0")
            text_embeddings = embeddings.embed_documents(["Hello world"])
    
    """
    
    model: str = "embed-english-v3.0"
    """Model name to use."""
    
    cohere_api_key: Optional[str] = None
    """Cohere API key. If not provided, will read from environment variable COHERE_API_KEY."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        raise NotImplementedError(
            "This is a placeholder class. Install langchain-cohere to use: pip install langchain-cohere"
        )
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        raise NotImplementedError(
            "This is a placeholder class. Install langchain-cohere to use: pip install langchain-cohere"
        )