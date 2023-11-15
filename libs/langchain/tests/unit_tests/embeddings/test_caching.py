"""Embeddings tests."""
from typing import List

import pytest

from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema.embeddings import Embeddings
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
def cache_embeddings() -> CacheBackedEmbeddings:
    """Create a cache backed embeddings."""
    store = InMemoryStore()
    embeddings = MockEmbeddings()
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace="test_namespace"
    )


def test_embed_documents(cache_embeddings: CacheBackedEmbeddings) -> None:
    texts = ["1", "22", "a", "333"]
    vectors = cache_embeddings.embed_documents(texts)
    expected_vectors: List[List[float]] = [[1, 2.0], [2.0, 3.0], [1.0, 2.0], [3.0, 4.0]]
    assert vectors == expected_vectors
    keys = list(cache_embeddings.document_embedding_store.yield_keys())
    assert len(keys) == 4
    # UUID is expected to be the same for the same text
    assert keys[0] == "test_namespace812b86c1-8ebf-5483-95c6-c95cf2b52d12"


def test_embed_query(cache_embeddings: CacheBackedEmbeddings) -> None:
    text = "query_text"
    vector = cache_embeddings.embed_query(text)
    expected_vector = [5.0, 6.0]
    assert vector == expected_vector
