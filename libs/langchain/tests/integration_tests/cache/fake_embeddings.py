"""Fake Embedding class for testing purposes."""

import math

from langchain_core.embeddings import Embeddings
from typing_extensions import override

fake_texts = ["foo", "bar", "baz"]


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return simple embeddings.

        Embeddings encode each text as its index.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return [[1.0] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    @override
    def embed_query(self, text: str) -> list[float]:
        """Return constant query embeddings.

        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return [1.0] * 9 + [0.0]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class ConsistentFakeEmbeddings(FakeEmbeddings):
    """Consistent fake embeddings.

    Fake embeddings which remember all the texts seen so far to return consistent
    vectors for the same texts.
    """

    def __init__(self, dimensionality: int = 10) -> None:
        self.known_texts: list[str] = []
        self.dimensionality = dimensionality

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            vector = [1.0] * (self.dimensionality - 1) + [
                float(self.known_texts.index(text)),
            ]
            out_vectors.append(vector)
        return out_vectors

    @override
    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return self.embed_documents([text])[0]


class AngularTwoDimensionalEmbeddings(Embeddings):
    """From angles (as strings in units of pi) to unit embedding vectors on a circle."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Make a list of texts into a list of embedding vectors."""
        return [self.embed_query(text) for text in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Convert input text to a 'vector' (list of floats).
        If the text is a number, use it as the angle for the
        unit vector in units of pi.
        Any other input text becomes the singular result [0, 0] !

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        try:
            angle = float(text)
            return [math.cos(angle * math.pi), math.sin(angle * math.pi)]
        except ValueError:
            # Assume: just test string, no attention is paid to values.
            return [0.0, 0.0]
