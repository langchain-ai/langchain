"""Interface for embedding models."""
from abc import ABC, abstractmethod
from typing import List


class Embeddings(ABC):
    """Interface for embedding models."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """DEPRECATED. Kept for backwards compatibility."""
        return self.embed_texts(texts)

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed search texts."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
