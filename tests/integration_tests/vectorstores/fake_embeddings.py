"""Fake Embedding class for testing purposes."""
from typing import List

from langchain.embeddings.base import Embeddings

fake_texts = ["foo", "bar", "baz"]


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index."""
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError()

    def embed_query(self, text: str) -> List[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]

    async def aembed_query(self, text: str) -> List[float]:
        raise NotImplementedError()


class ConsistentFakeEmbeddings(FakeEmbeddings):
    """Fake embeddings which remember all the texts seen so far to return consistent
    vectors for the same texts."""

    def __init__(self) -> None:
        self.known_texts: List[str] = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            vector = [float(1.0)] * 9 + [float(self.known_texts.index(text))]
            out_vectors.append(vector)
        return out_vectors

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError()

    def embed_query(self, text: str) -> List[float]:
        """Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown."""
        if text not in self.known_texts:
            return [float(1.0)] * 9 + [float(0.0)]
        return [float(1.0)] * 9 + [float(self.known_texts.index(text))]

    async def aembed_query(self, text: str) -> List[float]:
        raise NotImplementedError()
