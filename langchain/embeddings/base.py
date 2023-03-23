"""Interface for embedding models."""
from abc import ABC, abstractmethod
from pydantic import BaseModel, Extra, Field
from typing import List, Optional

from langchain.cache import BaseEmbeddingsCache

class Embeddings(BaseModel, ABC):
    """Interface for embedding models."""

    embeddings_cache: Optional[BaseEmbeddingsCache] = Field()
    """Cache for embeddings."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def _embed_query(self, text: str) -> List[float]:
        """Embed query text."""
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        if self.embeddings_cache is not None:
            cached = self.embeddings_cache.lookup(text)
            if cached is not None:
                return cached
        embedding = self._embed_query(text)
        if self.embeddings_cache is not None:
            self.embeddings_cache.update(text, embedding)
        return embedding
