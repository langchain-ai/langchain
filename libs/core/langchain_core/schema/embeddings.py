from abc import ABC, abstractmethod
from typing import List


class EmbeddingsInterface(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""

    @abstractmethod
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""

    @abstractmethod
    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
