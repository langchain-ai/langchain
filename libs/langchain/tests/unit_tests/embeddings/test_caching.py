"""Embeddings tests."""
from typing import List

import pytest

from langchain.embeddings import CacheBackedEmbedder
from langchain.embeddings.base import Embeddings
from langchain.storage.in_memory import InMemoryStore


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Simulate embedding documents
        embeddings: List[List[float]] = []
        for text in texts:
            embeddings.append([len(text), len(text) + 1])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # Simulate embedding a query
        return [5.0, 6.0]


@pytest.fixture
def cache_embedder() -> CacheBackedEmbedder:
    """Create a cache backed embedder."""
    store = InMemoryStore()
    embedder = MockEmbeddings()
    return CacheBackedEmbedder.from_bytes_store(
        embedder, store, namespace="test_namespace"
    )


def test_embed_documents(cache_embedder: CacheBackedEmbedder) -> None:
    texts = ["1", "22", "a", "333"]
    vectors = cache_embedder.embed_documents(texts)
    expected_vectors: List[List[float]] = [[1, 2.0], [2.0, 3.0], [1.0, 2.0], [3.0, 4.0]]
    assert vectors == expected_vectors
    keys = list(cache_embedder.document_embedding_cache.yield_keys())
    assert len(keys) == 4
    # UUID is expected to be the same for the same text
    assert keys[0] == "test_namespace812b86c1-8ebf-5483-95c6-c95cf2b52d12"


def test_embed_query(cache_embedder: CacheBackedEmbedder) -> None:
    text = "query_text"
    vector = cache_embedder.embed_query(text)
    expected_vector = [5.0, 6.0]
    assert vector == expected_vector
