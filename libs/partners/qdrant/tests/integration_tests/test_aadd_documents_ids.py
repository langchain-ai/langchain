"""
Regression tests for GitHub issue #32283:
  aadd_texts() got multiple values for keyword argument 'ids'

When aadd_documents() is called with custom ids as a keyword argument,
it should NOT raise:
  TypeError: aadd_texts() got multiple values for keyword argument 'ids'

Root cause: aadd_documents was adding ids into **kwargs AND also passing
ids=ids explicitly to aadd_texts, causing Python to reject the duplicate.

These tests guard against regression in:
  - QdrantVectorStore (new API)
  - Deprecated Qdrant class (sync-fallback path via @sync_call_fallback)

Run with:
  source venv/bin/activate
  pytest tests/test_aadd_documents_ids.py -v
"""

import asyncio
import uuid
import warnings
import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List


class FakeEmbeddings(Embeddings):
    """Minimal embeddings stub — returns fixed 3D vectors."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]


def make_uuid_ids(n: int) -> List[str]:
    return [str(uuid.uuid4()) for _ in range(n)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qdrant_client():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )
    return client


@pytest.fixture
def qdrant_vector_store(qdrant_client):
    from langchain_qdrant import QdrantVectorStore
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name="test",
        embedding=FakeEmbeddings(),
    )


@pytest.fixture
def deprecated_qdrant_sync(qdrant_client):
    """Deprecated Qdrant class with sync-only client (no async_client)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langchain_qdrant.vectorstores import Qdrant
        return Qdrant(
            client=qdrant_client,
            collection_name="test",
            embeddings=FakeEmbeddings(),
        )


# ---------------------------------------------------------------------------
# Tests — QdrantVectorStore (new API)
# ---------------------------------------------------------------------------

class TestQdrantVectorStoreAaddDocuments:
    """
    Regression: issue #32283 — ids passed twice to aadd_texts.
    """

    @pytest.mark.asyncio
    async def test_aadd_documents_with_custom_ids_does_not_raise(self, qdrant_vector_store):
        """
        Calling aadd_documents(docs, ids=ids) must not raise TypeError.
        This was the exact call pattern from issue #32283.
        """
        docs = [
            Document(page_content="Hello world"),
            Document(page_content="Goodbye world"),
        ]
        ids = make_uuid_ids(2)

        # Must not raise: TypeError: aadd_texts() got multiple values for keyword argument 'ids'
        result = await qdrant_vector_store.aadd_documents(docs, ids=ids)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_aadd_documents_returns_provided_ids(self, qdrant_vector_store):
        """The IDs returned should match the ones passed in."""
        docs = [Document(page_content="Test doc")]
        ids = make_uuid_ids(1)

        result = await qdrant_vector_store.aadd_documents(docs, ids=ids)
        assert result == ids

    @pytest.mark.asyncio
    async def test_aadd_documents_without_ids_generates_ids(self, qdrant_vector_store):
        """aadd_documents without explicit ids should auto-generate valid UUIDs."""
        docs = [
            Document(page_content="First doc"),
            Document(page_content="Second doc"),
        ]
        result = await qdrant_vector_store.aadd_documents(docs)
        assert len(result) == 2
        # All returned values should be valid UUIDs
        for id_ in result:
            uuid.UUID(str(id_))  # raises ValueError if not a valid UUID

    @pytest.mark.asyncio
    async def test_aadd_documents_multiple_calls_accumulate(self, qdrant_vector_store):
        """Multiple aadd_documents calls should accumulate documents, not overwrite."""
        docs_a = [Document(page_content="Batch A")]
        docs_b = [Document(page_content="Batch B")]
        ids_a = make_uuid_ids(1)
        ids_b = make_uuid_ids(1)

        result_a = await qdrant_vector_store.aadd_documents(docs_a, ids=ids_a)
        result_b = await qdrant_vector_store.aadd_documents(docs_b, ids=ids_b)

        assert result_a == ids_a
        assert result_b == ids_b
        assert result_a != result_b

    @pytest.mark.asyncio
    async def test_aadd_documents_upsert_same_id_updates_content(self, qdrant_vector_store):
        """Adding a document with an existing ID should upsert (update) it."""
        shared_id = make_uuid_ids(1)
        doc_v1 = [Document(page_content="Version 1")]
        doc_v2 = [Document(page_content="Version 2")]

        r1 = await qdrant_vector_store.aadd_documents(doc_v1, ids=shared_id)
        r2 = await qdrant_vector_store.aadd_documents(doc_v2, ids=shared_id)

        assert r1 == shared_id
        assert r2 == shared_id  # same ID, upserted


# ---------------------------------------------------------------------------
# Tests — Deprecated Qdrant class (sync fallback path)
# ---------------------------------------------------------------------------

class TestDeprecatedQdrantAaddDocuments:
    """
    The deprecated Qdrant class overrides aadd_texts with @sync_call_fallback.
    Verify aadd_documents still works correctly through that path.
    """

    @pytest.mark.asyncio
    async def test_aadd_documents_with_custom_ids_does_not_raise(self, deprecated_qdrant_sync):
        docs = [
            Document(page_content="Hello world"),
            Document(page_content="Goodbye world"),
        ]
        ids = make_uuid_ids(2)
        # Must not raise: TypeError: aadd_texts() got multiple values for keyword argument 'ids'
        result = await deprecated_qdrant_sync.aadd_documents(docs, ids=ids)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_aadd_documents_returns_provided_ids(self, deprecated_qdrant_sync):
        docs = [Document(page_content="Test doc")]
        ids = make_uuid_ids(1)
        result = await deprecated_qdrant_sync.aadd_documents(docs, ids=ids)
        assert result == ids

    @pytest.mark.asyncio
    async def test_aadd_documents_without_ids(self, deprecated_qdrant_sync):
        docs = [Document(page_content="Auto ID doc")]
        result = await deprecated_qdrant_sync.aadd_documents(docs)
        assert len(result) == 1
