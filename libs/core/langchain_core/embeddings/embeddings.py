"""**Embeddings** interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from langchain_core.rate_limiters import BaseRateLimiter


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
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronous Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return await run_in_executor(None, self.embed_query, text)

    def with_rate_limiter(self, rate_limiter: BaseRateLimiter) -> Embeddings:
        """Create a rate-limited wrapper around this embeddings instance.

        The returned wrapper acquires the rate limiter before each embedding
        call, throttling requests to stay within configured limits. This is
        consistent with the rate limiting support on chat models.

        Args:
            rate_limiter: The rate limiter to use (e.g., `InMemoryRateLimiter`).

        Returns:
            A new `Embeddings` instance that rate-limits all calls.

        Examples:
            ```python
            from langchain_core.embeddings import FakeEmbeddings
            from langchain_core.rate_limiters import InMemoryRateLimiter

            embeddings = FakeEmbeddings(size=10).with_rate_limiter(
                InMemoryRateLimiter(requests_per_second=5, max_bucket_size=10)
            )
            embeddings.embed_documents(["hello", "world"])
            ```
        """
        return _RateLimitedEmbeddings(self, rate_limiter)


class _RateLimitedEmbeddings(Embeddings):
    """Transparent wrapper that rate-limits all embedding calls.

    This class is not part of the public API. Use
    ``Embeddings.with_rate_limiter()`` to create instances.
    """

    def __init__(self, base: Embeddings, rate_limiter: BaseRateLimiter) -> None:
        self._base = base
        self._rate_limiter = rate_limiter

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Rate-limit then delegate to the underlying embeddings.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        self._rate_limiter.acquire(blocking=True)
        return self._base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Rate-limit then delegate to the underlying embeddings.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        self._rate_limiter.acquire(blocking=True)
        return self._base.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async rate-limit then delegate to the underlying embeddings.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        await self._rate_limiter.aacquire(blocking=True)
        return await self._base.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Async rate-limit then delegate to the underlying embeddings.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        await self._rate_limiter.aacquire(blocking=True)
        return await self._base.aembed_query(text)
