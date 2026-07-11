"""Regression tests for document add methods with custom IDs.

These tests verify the fix for issue #32283 where aadd_documents()
raised TypeError when passing custom document IDs via the ids= kwarg.
The fix added a guard to check if "ids" is already in kwargs before
adding it, preventing the "multiple values for keyword argument 'ids'"
error.
"""

import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.vectorstores import Qdrant  # deprecated class


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]


def _embedding_function(text: str) -> list[float]:
    """Callable embedding function for deprecated Qdrant class."""
    return [1.0, 2.0, 3.0]


class TestAddDocumentsWithCustomIds:
    """Test that adding documents with custom IDs works correctly."""

    @pytest.fixture
    def client(self):
        """Create a QdrantClient with an in-memory collection."""
        client = QdrantClient(location=":memory:")
        collection_name = f"test_custom_ids_{uuid.uuid4().hex}"
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
        )
        return client, collection_name

    @pytest.fixture
    def deprecated_collection(self):
        """Create a QdrantClient with a collection for the deprecated Qdrant class."""
        client = QdrantClient(location=":memory:")
        collection_name = f"test_deprecated_{uuid.uuid4().hex}"
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
        )
        return client, collection_name

    @pytest.mark.asyncio
    async def test_qdrant_vectorstore_aadd_documents_with_custom_ids(
        self, client
    ) -> None:
        """Test QdrantVectorStore.aadd_documents with custom ids kwarg.

        This is a regression test for issue #32283.
        Previously, passing ids= kwarg alongside documents raised:
            TypeError: aadd_texts() got multiple values for keyword argument 'ids'
        """
        qclient, collection_name = client
        store = QdrantVectorStore(
            client=qclient,
            collection_name=collection_name,
            embedding=MockEmbeddings(),
        )

        documents = [
            Document(page_content="Hello world"),
            Document(page_content="Goodbye world"),
        ]
        custom_ids = [uuid.uuid4().hex, uuid.uuid4().hex]

        # This should NOT raise TypeError
        result_ids = await store.aadd_documents(documents, ids=custom_ids)

        assert result_ids == custom_ids

    @pytest.mark.asyncio
    async def test_qdrant_vectorstore_aadd_documents_without_custom_ids(
        self, client
    ) -> None:
        """Test that aadd_documents still works without custom ids."""
        qclient, collection_name = client
        store = QdrantVectorStore(
            client=qclient,
            collection_name=collection_name,
            embedding=MockEmbeddings(),
        )

        documents = [
            Document(page_content="Hello world"),
            Document(page_content="Goodbye world"),
        ]

        # Should work without ids kwarg
        result_ids = await store.aadd_documents(documents)
        assert len(result_ids) == 2

    @pytest.mark.asyncio
    async def test_qdrant_vectorstore_add_documents_sync_with_custom_ids(
        self, client
    ) -> None:
        """Test sync add_documents with custom ids kwarg.

        Regression test for the same bug in the sync path.
        """
        qclient, collection_name = client
        store = QdrantVectorStore(
            client=qclient,
            collection_name=collection_name,
            embedding=MockEmbeddings(),
        )

        documents = [
            Document(page_content="Hello world"),
            Document(page_content="Goodbye world"),
        ]
        custom_ids = [uuid.uuid4().hex, uuid.uuid4().hex]

        # This should NOT raise TypeError
        result_ids = store.add_documents(documents, ids=custom_ids)

        assert result_ids == custom_ids

    @pytest.mark.asyncio
    async def test_qdrant_deprecated_class_aadd_documents_with_custom_ids(
        self, deprecated_collection
    ) -> None:
        """Test deprecated Qdrant class aadd_documents with custom ids.

        Regression test for issue #32283.
        """
        qclient, collection_name = deprecated_collection
        # Use embedding_function for deprecated class (embeddings as callable)
        store = Qdrant(
            client=qclient,
            collection_name=collection_name,
            embeddings=_embedding_function,
        )

        documents = [
            Document(page_content="Hello world"),
            Document(page_content="Goodbye world"),
        ]
        custom_ids = [uuid.uuid4().hex, uuid.uuid4().hex]

        # This should NOT raise TypeError
        result_ids = await store.aadd_documents(documents, ids=custom_ids)

        assert result_ids == custom_ids

    def test_qdrant_deprecated_class_add_documents_sync_with_custom_ids(
        self, deprecated_collection
    ) -> None:
        """Test deprecated Qdrant class sync add_documents with custom ids.

        Regression test for the same bug in the sync path.
        """
        qclient, collection_name = deprecated_collection
        # Use embedding_function for deprecated class (embeddings as callable)
        store = Qdrant(
            client=qclient,
            collection_name=collection_name,
            embeddings=_embedding_function,
        )

        documents = [
            Document(page_content="Hello world"),
            Document(page_content="Goodbye world"),
        ]
        custom_ids = [uuid.uuid4().hex, uuid.uuid4().hex]

        # This should NOT raise TypeError
        result_ids = store.add_documents(documents, ids=custom_ids)

        assert result_ids == custom_ids
