"""Interface for embedding models."""
from abc import ABC, abstractmethod
from typing import List


class TextEmbeddingModel(ABC):
    """Interface for text embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""


# For backwards compatibility.
Embedding = TextEmbeddingModel
