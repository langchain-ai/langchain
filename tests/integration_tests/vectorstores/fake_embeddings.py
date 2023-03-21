"""Fake Embedding class for testing purposes."""
from typing import List

from langchain.embeddings.base import Embeddings

fake_texts = ["foo", "bar", "baz"]


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings which encode each text as its index
        in the input list."""
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        """Return simple query embedding matching the embedding of the 
        first text of any list passed to embed_documents. Computed distance 
        to any embedded document will be the index of that document as it was 
        passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]
