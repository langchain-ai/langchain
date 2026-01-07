"""**Embeddings** interface."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.runnables.config import run_in_executor


class Embeddings(ABC):
    """Interface for embedding models.

    This is an interface meant for implementing text embedding models.

    Text embedding models are used to map text to a vector (a point in n-dimensional
    space).

    Texts that are similar will usually be mapped to points that are close to each
    other in this space. The exact details of what's considered "similar" and how
    "distance" is measured in this space are dependent on the specific embedding model.

    This abstraction contains a method for embedding a list of documents and a method
    for embedding a query text. The embedding of a query text is expected to be a single
    vector, while the embedding of a list of documents is expected to be a list of
    vectors.

    Usually the query embedding is identical to the document embedding, but the
    abstraction allows treating them independently.

    In addition to the synchronous methods, this interface also provides asynchronous
    versions of the methods.

    By default, the asynchronous methods are implemented using the synchronous methods;
    however, implementations may choose to override the asynchronous methods with
    an async native implementation for performance reasons.
    """

    @abstractmethod
    def embed_documents(
        self, texts: list[str], **kwargs: Any,
    ) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.
            output_dimensionality: Embeddings dimension.

        Returns:
            List of embeddings.
        """

    @abstractmethod
    def embed_query(
        self, text: str, **kwargs: Any,
    ) -> list[float]:
        """Embed query text.

        Args:
            text: Text to embed.
            output_dimensionality: Embeddings dimension.

        Returns:
            Embedding.
        """

    async def aembed_documents(
        self, texts: list[str], **kwargs: Any,
    ) -> list[list[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.
            output_dimensionality: Embeddings dimension.

        Returns:
            List of embeddings.
        """
        return await run_in_executor(
            None, self.embed_documents, texts, **kwargs
        )

    async def aembed_query(
        self, text: str, **kwargs: Any,
    ) -> list[float]:
        """Asynchronous Embed query text.

        Args:
            text: Text to embed.
            output_dimensionality: Embeddings dimension.

        Returns:
            Embedding.
        """
        return await run_in_executor(
            None, self.embed_query, text, **kwargs
        )
