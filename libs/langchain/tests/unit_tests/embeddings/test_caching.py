"""Embeddings tests."""

import pytest
from langchain_core.embeddings import Embeddings

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage.in_memory import InMemoryStore


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Simulate embedding documents
        embeddings: list[list[float]] = []
        for text in texts:
            if text == "RAISE_EXCEPTION":
                raise ValueError("Simulated embedding failure")
            embeddings.append([len(text), len(text) + 1])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
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


@pytest.fixture
def cache_embeddings_batch() -> CacheBackedEmbeddings:
    """Create a cache backed embeddings with a batch_size of 3."""
    store = InMemoryStore()
    embeddings = MockEmbeddings()
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace="test_namespace", batch_size=3
    )


@pytest.fixture
def cache_embeddings_with_query() -> CacheBackedEmbeddings:
    """Create a cache backed embeddings with query caching."""
    doc_store = InMemoryStore()
    query_store = InMemoryStore()
    embeddings = MockEmbeddings()
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        document_embedding_cache=doc_store,
        namespace="test_namespace",
        query_embedding_cache=query_store,
    )


def test_embed_documents(cache_embeddings: CacheBackedEmbeddings) -> None:
    texts = ["1", "22", "a", "333"]
    vectors = cache_embeddings.embed_documents(texts)
    expected_vectors: list[list[float]] = [[1, 2.0], [2.0, 3.0], [1.0, 2.0], [3.0, 4.0]]
    assert vectors == expected_vectors
    keys = list(cache_embeddings.document_embedding_store.yield_keys())
    assert len(keys) == 4
    # UUID is expected to be the same for the same text
    assert keys[0] == "test_namespace812b86c1-8ebf-5483-95c6-c95cf2b52d12"


def test_embed_documents_batch(cache_embeddings_batch: CacheBackedEmbeddings) -> None:
    # "RAISE_EXCEPTION" forces a failure in batch 2
    texts = ["1", "22", "a", "333", "RAISE_EXCEPTION"]
    try:
        cache_embeddings_batch.embed_documents(texts)
    except ValueError:
        pass
    keys = list(cache_embeddings_batch.document_embedding_store.yield_keys())
    # only the first batch of three embeddings should exist
    assert len(keys) == 3
    # UUID is expected to be the same for the same text
    assert keys[0] == "test_namespace812b86c1-8ebf-5483-95c6-c95cf2b52d12"


def test_embed_query(cache_embeddings: CacheBackedEmbeddings) -> None:
    text = "query_text"
    vector = cache_embeddings.embed_query(text)
    expected_vector = [5.0, 6.0]
    assert vector == expected_vector
    assert cache_embeddings.query_embedding_store is None


def test_embed_cached_query(cache_embeddings_with_query: CacheBackedEmbeddings) -> None:
    text = "query_text"
    vector = cache_embeddings_with_query.embed_query(text)
    expected_vector = [5.0, 6.0]
    assert vector == expected_vector
    keys = list(cache_embeddings_with_query.query_embedding_store.yield_keys())  # type: ignore[union-attr]
    assert len(keys) == 1
    assert keys[0] == "test_namespace89ec3dae-a4d9-5636-a62e-ff3b56cdfa15"


async def test_aembed_documents(cache_embeddings: CacheBackedEmbeddings) -> None:
    texts = ["1", "22", "a", "333"]
    vectors = await cache_embeddings.aembed_documents(texts)
    expected_vectors: list[list[float]] = [[1, 2.0], [2.0, 3.0], [1.0, 2.0], [3.0, 4.0]]
    assert vectors == expected_vectors
    keys = [
        key async for key in cache_embeddings.document_embedding_store.ayield_keys()
    ]
    assert len(keys) == 4
    # UUID is expected to be the same for the same text
    assert keys[0] == "test_namespace812b86c1-8ebf-5483-95c6-c95cf2b52d12"


async def test_aembed_documents_batch(
    cache_embeddings_batch: CacheBackedEmbeddings,
) -> None:
    # "RAISE_EXCEPTION" forces a failure in batch 2
    texts = ["1", "22", "a", "333", "RAISE_EXCEPTION"]
    try:
        await cache_embeddings_batch.aembed_documents(texts)
    except ValueError:
        pass
    keys = [
        key
        async for key in cache_embeddings_batch.document_embedding_store.ayield_keys()
    ]
    # only the first batch of three embeddings should exist
    assert len(keys) == 3
    # UUID is expected to be the same for the same text
    assert keys[0] == "test_namespace812b86c1-8ebf-5483-95c6-c95cf2b52d12"


async def test_aembed_query(cache_embeddings: CacheBackedEmbeddings) -> None:
    text = "query_text"
    vector = await cache_embeddings.aembed_query(text)
    expected_vector = [5.0, 6.0]
    assert vector == expected_vector


async def test_aembed_query_cached(
    cache_embeddings_with_query: CacheBackedEmbeddings,
) -> None:
    text = "query_text"
    await cache_embeddings_with_query.aembed_query(text)
    keys = list(cache_embeddings_with_query.query_embedding_store.yield_keys())  # type: ignore[union-attr]
    assert len(keys) == 1
    assert keys[0] == "test_namespace89ec3dae-a4d9-5636-a62e-ff3b56cdfa15"
