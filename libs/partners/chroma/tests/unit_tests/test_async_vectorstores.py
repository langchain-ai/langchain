"""Unit tests for async functionality in Chroma vector store."""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings

from langchain_chroma.vectorstores import Chroma


class AsyncFakeEmbeddings(FakeEmbeddings):
    """Fake embeddings with async support for testing."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_documents."""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Async version of embed_query."""
        return self.embed_query(text)


class MockAsyncCollection:
    """Mock async collection for testing."""

    def __init__(self, name: str):
        self.name = name
        self.data: dict[str, Any] = {}

    async def upsert(
        self,
        ids: list[str],
        documents: Optional[list[str]] = None,
        embeddings: Optional[list[list[float]]] = None,
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> None:
        """Mock upsert method."""
        for i, id_ in enumerate(ids):
            self.data[id_] = {
                "document": documents[i] if documents else None,
                "embedding": embeddings[i] if embeddings else None,
                "metadata": metadatas[i] if metadatas else None,
            }

    async def query(
        self,
        query_texts: Optional[list[str]] = None,
        query_embeddings: Optional[list[list[float]]] = None,
        n_results: int = 4,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        **kwargs: Any,
    ) -> dict:
        """Mock query method."""
        # Return mock results
        ids = list(self.data.keys())[:n_results]
        documents = [self.data[id_]["document"] for id_ in ids]
        metadatas = [self.data[id_]["metadata"] for id_ in ids]
        distances = [[0.1 * i for i in range(len(ids))]]

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": distances,
            "embeddings": None,
        }

    async def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> None:
        """Mock delete method."""
        if ids:
            for id_ in ids:
                self.data.pop(id_, None)


class MockAsyncClient:
    """Mock async client for testing."""

    def __init__(self):
        self.collections: dict[str, MockAsyncCollection] = {}

    async def get_or_create_collection(
        self,
        name: str,
        embedding_function: Any = None,
        metadata: Optional[dict] = None,
        configuration: Optional[dict] = None,
    ) -> MockAsyncCollection:
        """Mock get_or_create_collection method."""
        if name not in self.collections:
            self.collections[name] = MockAsyncCollection(name)
        return self.collections[name]

    async def delete_collection(self, name: str) -> None:
        """Mock delete_collection method."""
        self.collections.pop(name, None)


@pytest.mark.asyncio
async def test_async_client_initialization() -> None:
    """Test that Chroma can be initialized with an async client."""
    async_client = MockAsyncClient()

    # Initialize with async client only
    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    assert chroma._async_client is not None
    assert chroma._client is None
    assert chroma._async_initialized is False


@pytest.mark.asyncio
async def test_async_collection_initialization() -> None:
    """Test async collection initialization."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    # Collection should be initialized on first async operation
    await chroma._aensure_collection()

    assert chroma._async_initialized is True
    assert hasattr(chroma, "_async_chroma_collection")


@pytest.mark.asyncio
async def test_aadd_texts() -> None:
    """Test async add_texts functionality."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]

    # Add texts asynchronously
    ids = await chroma.aadd_texts(texts=texts, metadatas=metadatas)

    assert len(ids) == 3
    assert chroma._async_initialized is True


@pytest.mark.asyncio
async def test_aadd_documents() -> None:
    """Test async add_documents functionality."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    documents = [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
        Document(page_content="baz", metadata={"page": "2"}),
    ]

    # Add documents asynchronously
    ids = await chroma.aadd_documents(documents=documents)

    assert len(ids) == 3
    assert chroma._async_initialized is True


@pytest.mark.asyncio
async def test_asimilarity_search() -> None:
    """Test async similarity search."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    # Add some texts first
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    await chroma.aadd_texts(texts=texts, metadatas=metadatas)

    # Perform similarity search
    results = await chroma.asimilarity_search("foo", k=2)

    assert len(results) <= 2
    assert all(isinstance(doc, Document) for doc in results)


@pytest.mark.asyncio
async def test_asimilarity_search_with_score() -> None:
    """Test async similarity search with scores."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    # Add some texts first
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    await chroma.aadd_texts(texts=texts, metadatas=metadatas)

    # Perform similarity search with scores
    results = await chroma.asimilarity_search_with_score("foo", k=2)

    assert len(results) <= 2
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)


@pytest.mark.asyncio
async def test_adelete() -> None:
    """Test async delete functionality."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    # Add some texts first
    texts = ["foo", "bar", "baz"]
    ids = await chroma.aadd_texts(texts=texts)

    # Delete specific IDs
    await chroma.adelete(ids=[ids[0]])

    # Verify deletion worked (this would need actual verification in real tests)
    assert chroma._async_initialized is True


@pytest.mark.asyncio
async def test_adelete_collection() -> None:
    """Test async delete collection functionality."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    # Initialize collection
    await chroma._aensure_collection()
    assert chroma._async_initialized is True

    # Delete collection
    await chroma.adelete_collection()

    assert chroma._async_initialized is False
    assert chroma._async_chroma_collection is None


@pytest.mark.asyncio
async def test_areset_collection() -> None:
    """Test async reset collection functionality."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    # Add some texts
    texts = ["foo", "bar", "baz"]
    await chroma.aadd_texts(texts=texts)

    # Reset collection
    await chroma.areset_collection()

    # Collection should be re-initialized but empty
    assert chroma._async_initialized is True


@pytest.mark.asyncio
async def test_error_without_async_client() -> None:
    """Test that async methods raise errors when async_client is not provided."""
    # Initialize without async client
    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(size=10),
    )

    # All async methods should raise ValueError
    with pytest.raises(ValueError, match="async_client"):
        await chroma.aadd_texts(["test"])

    with pytest.raises(ValueError, match="async_client"):
        await chroma.asimilarity_search("test")

    with pytest.raises(ValueError, match="async_client"):
        await chroma.asimilarity_search_with_score("test")

    with pytest.raises(ValueError, match="async_client"):
        await chroma.adelete(["test_id"])

    with pytest.raises(ValueError, match="async_client"):
        await chroma.adelete_collection()

    with pytest.raises(ValueError, match="async_client"):
        await chroma.areset_collection()


@pytest.mark.asyncio
async def test_sync_methods_error_with_only_async_client() -> None:
    """Test that sync methods raise errors when only async_client is provided."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(size=10),
        async_client=async_client,
    )

    # Sync methods that require collection should raise ValueError
    with pytest.raises(ValueError, match="async_client"):
        _ = chroma._collection  # Accessing sync collection property

    with pytest.raises(ValueError, match="async_client"):
        chroma.add_texts(["test"])


def test_both_sync_and_async_clients() -> None:
    """Test that both sync and async clients can be provided."""
    sync_client = MagicMock()
    sync_client.get_or_create_collection.return_value = MagicMock()

    async_client = MockAsyncClient()

    # Initialize with both clients
    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(size=10),
        client=sync_client,
        async_client=async_client,
    )

    assert chroma._client is not None
    assert chroma._async_client is not None

    # Sync collection should be initialized
    assert chroma._chroma_collection is not None


@pytest.mark.asyncio
async def test_async_with_metadata_filtering() -> None:
    """Test async operations with metadata filtering."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    # Add texts with metadata
    texts = ["foo", "bar", "baz", "qux"]
    metadatas = [
        {"type": "a", "page": "0"},
        {"type": "b", "page": "1"},
        {"type": "a", "page": "2"},
        {"type": "b", "page": "3"},
    ]
    await chroma.aadd_texts(texts=texts, metadatas=metadatas)

    # Search with filter
    results = await chroma.asimilarity_search("foo", k=2, filter={"type": "a"})

    assert len(results) <= 2


@pytest.mark.asyncio
async def test_async_empty_metadata_handling() -> None:
    """Test async operations with empty metadata."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    # Add texts with mixed empty and non-empty metadata
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": "0"}, {}, {"page": "2"}]

    ids = await chroma.aadd_texts(texts=texts, metadatas=metadatas)

    assert len(ids) == 3


@pytest.mark.asyncio
async def test_concurrent_async_operations() -> None:
    """Test that multiple async operations can run concurrently."""
    async_client = MockAsyncClient()

    chroma = Chroma(
        collection_name="test_collection",
        embedding_function=AsyncFakeEmbeddings(size=10),
        async_client=async_client,
    )

    # Add initial data
    await chroma.aadd_texts(["initial"])

    # Run multiple operations concurrently
    tasks = [
        chroma.aadd_texts(["text1"]),
        chroma.aadd_texts(["text2"]),
        chroma.asimilarity_search("initial"),
    ]

    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    assert all(results[0])  # First add_texts returned IDs
    assert all(results[1])  # Second add_texts returned IDs
    assert isinstance(results[2], list)  # Search returned documents
