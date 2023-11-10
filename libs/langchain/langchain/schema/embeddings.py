import asyncio
from abc import ABC, abstractmethod
from typing import List


class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.embed_documents, texts
        )

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.embed_query, text
        )

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        """Embed images.

        Args:
            uris (List[str]): List of image URIs.

        Returns:
            Optional[List[List[float]]]: List of image embeddings or None
        """
        # Default implementation could return None or raise NotImplementedError
        # return None
        raise NotImplementedError("embed_image method is not implemented.")
