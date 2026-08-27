"""Embeddings tests."""

import contextlib
import hashlib
import importlib
import warnings

import pytest
from langchain_core.embeddings import Embeddings
from typing_extensions import override

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage.in_memory import InMemoryStore


class MockEmbeddings(Embeddings):
    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Simulate embedding documents
        embeddings: list[list[float]] = []
        for text in texts:
            if text == "RAISE_EXCEPTION":
                msg = "Simulated embedding failure"
                raise ValueError(msg)
            embeddings.append([len(text), len(text) + 1])
        return embeddings

    @override
    def embed_query(self, text: str) -> list[float]:
        # Simulate embedding a query
        return [5.0, 6.0]


@pytest.fixture
def cache_embeddings() -> CacheBackedEmbeddings:
    """Create a cache backed embeddings."""
    store = InMemoryStore()
    embeddings = MockEmbeddings()
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        store,
        namespace="test_namespace",
    )


@pytest.fixture
def cache_embeddings_batch() -> CacheBackedEmbeddings:
    """Create a cache backed embeddings with a batch_size of 3."""
    store = InMemoryStore()
    embeddings = MockEmbeddings()
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        store,
        namespace="test_namespace",
        batch_size=3,
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
    with contextlib.suppress(ValueError):
        cache_embeddings_batch.embed_documents(texts)
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
    with contextlib.suppress(ValueError):
        await cache_embeddings_batch.aembed_documents(texts)
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


def test_blake2b_encoder() -> None:
    """Test that the blake2b encoder is used to encode keys in the cache store."""
    store = InMemoryStore()
    emb = MockEmbeddings()
    cbe = CacheBackedEmbeddings.from_bytes_store(
        emb,
        store,
        namespace="ns_",
        key_encoder="blake2b",
    )

    text = "blake"
    cbe.embed_documents([text])

    # rebuild the key exactly as the library does
    expected_key = "ns_" + hashlib.blake2b(text.encode()).hexdigest()
    assert list(cbe.document_embedding_store.yield_keys()) == [expected_key]


def test_sha256_encoder() -> None:
    """Test that the sha256 encoder is used to encode keys in the cache store."""
    store = InMemoryStore()
    emb = MockEmbeddings()
    cbe = CacheBackedEmbeddings.from_bytes_store(
        emb,
        store,
        namespace="ns_",
        key_encoder="sha256",
    )

    text = "foo"
    cbe.embed_documents([text])

    # rebuild the key exactly as the library does
    expected_key = "ns_" + hashlib.sha256(text.encode()).hexdigest()
    assert list(cbe.document_embedding_store.yield_keys()) == [expected_key]


def test_sha512_encoder() -> None:
    """Test that the sha512 encoder is used to encode keys in the cache store."""
    store = InMemoryStore()
    emb = MockEmbeddings()
    cbe = CacheBackedEmbeddings.from_bytes_store(
        emb,
        store,
        namespace="ns_",
        key_encoder="sha512",
    )

    text = "foo"
    cbe.embed_documents([text])

    # rebuild the key exactly as the library does
    expected_key = "ns_" + hashlib.sha512(text.encode()).hexdigest()
    assert list(cbe.document_embedding_store.yield_keys()) == [expected_key]


def test_sha1_warning_emitted_once() -> None:
    """Test that a warning is emitted when using SHA-1 as the default key encoder."""
    module = importlib.import_module(CacheBackedEmbeddings.__module__)

    # Create a *temporary* MonkeyPatch object whose effects disappear
    # automatically when the with-block exits.
    with pytest.MonkeyPatch.context() as mp:
        # We're monkey patching the module to reset the `_warned_about_sha1` flag
        # which may have been set while testing other parts of the codebase.
        mp.setattr(module, "_warned_about_sha1", False, raising=False)

        store = InMemoryStore()
        emb = MockEmbeddings()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CacheBackedEmbeddings.from_bytes_store(emb, store)  # triggers warning
            CacheBackedEmbeddings.from_bytes_store(emb, store)  # silent

        sha1_msgs = [w for w in caught if "SHA-1" in str(w.message)]
        assert len(sha1_msgs) == 1


def test_custom_encoder() -> None:
    """Test that a custom encoder can be used to encode keys in the cache store."""
    store = InMemoryStore()
    emb = MockEmbeddings()

    def custom_upper(text: str) -> str:  # very simple demo encoder
        return "CUSTOM_" + text.upper()

    cbe = CacheBackedEmbeddings.from_bytes_store(emb, store, key_encoder=custom_upper)
    txt = "x"
    cbe.embed_documents([txt])

    assert list(cbe.document_embedding_store.yield_keys()) == ["CUSTOM_X"]
