"""Async delete and get_by_ids tests for QdrantVectorStore."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient, models

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    ConsistentFakeSparseEmbeddings,
)
from tests.integration_tests.fixtures import qdrant_locations, retrieval_modes


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
async def test_async_delete(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
) -> None:
    """Test async delete functionality."""
    texts = ["foo", "bar", "baz"]
    # Use UUID format for IDs (required by in-memory Qdrant)
    ids = [
        "fa38d572-4c31-4579-aedc-1960d79df6d1",
        "fa38d572-4c31-4579-aedc-1960d79df6d2",
        "fa38d572-4c31-4579-aedc-1960d79df6d3",
    ]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        ids=ids,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    # Verify initial count
    assert docsearch.client.count(docsearch.collection_name).count == 3

    # Delete one document
    result = await docsearch.adelete(ids=[ids[0]])
    assert result is True

    # Verify count after deletion
    assert docsearch.client.count(docsearch.collection_name).count == 2

    # Delete remaining documents
    result = await docsearch.adelete(ids=[ids[1], ids[2]])
    assert result is True

    # Verify all deleted
    assert docsearch.client.count(docsearch.collection_name).count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
async def test_async_get_by_ids(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
) -> None:
    """Test async get_by_ids functionality."""
    texts = ["foo", "bar", "baz"]
    # Use UUID format for IDs (required by in-memory Qdrant)
    ids = [
        "fa38d572-4c31-4579-aedc-1960d79df6d1",
        "fa38d572-4c31-4579-aedc-1960d79df6d2",
        "fa38d572-4c31-4579-aedc-1960d79df6d3",
    ]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        ids=ids,
        metadatas=metadatas,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    # Get all documents by ids
    docs = await docsearch.aget_by_ids(ids)
    assert len(docs) == 3

    # Verify content
    contents = {doc.page_content for doc in docs}
    assert contents == {"foo", "bar", "baz"}

    # Get subset of documents
    docs = await docsearch.aget_by_ids([ids[0], ids[2]])
    assert len(docs) == 2
    contents = {doc.page_content for doc in docs}
    assert contents == {"foo", "baz"}


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
async def test_async_delete_and_get_by_ids_with_int_ids(
    location: str,
    vector_name: str,
) -> None:
    """Test async delete and get_by_ids with integer IDs."""
    client = QdrantClient(location)
    collection_name = uuid.uuid4().hex
    vectors_config = {
        vector_name: models.VectorParams(size=10, distance=models.Distance.COSINE)
    }
    client.recreate_collection(collection_name, vectors_config=vectors_config)

    vec_store = QdrantVectorStore(
        client,
        collection_name,
        embedding=ConsistentFakeEmbeddings(),
        vector_name=vector_name,
    )

    # Add texts with integer IDs
    int_ids: list[str | int] = [1, 2, 3]
    await vec_store.aadd_texts(
        ["foo", "bar", "baz"],
        metadatas=[{"page": i} for i in range(3)],
        ids=int_ids,
    )

    # Verify documents were added
    assert client.count(collection_name).count == 3

    # Get by integer IDs
    docs = await vec_store.aget_by_ids(int_ids)
    assert len(docs) == 3

    # Delete by integer IDs
    result = await vec_store.adelete(ids=[1, 2])
    assert result is True
    assert client.count(collection_name).count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
async def test_async_get_by_ids_empty_list(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
) -> None:
    """Test async get_by_ids with empty list returns empty list."""
    texts = ["foo", "bar", "baz"]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    # Get with empty list should return empty list
    docs = await docsearch.aget_by_ids([])
    assert len(docs) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
async def test_async_get_by_ids_nonexistent(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
) -> None:
    """Test async get_by_ids with nonexistent IDs returns empty list."""
    texts = ["foo", "bar", "baz"]
    # Use UUID format for IDs (required by in-memory Qdrant)
    ids = [
        "fa38d572-4c31-4579-aedc-1960d79df6d1",
        "fa38d572-4c31-4579-aedc-1960d79df6d2",
        "fa38d572-4c31-4579-aedc-1960d79df6d3",
    ]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        ids=ids,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    # Get with nonexistent IDs should return empty list
    # Use UUID format for nonexistent IDs too
    nonexistent_ids = [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]
    docs = await docsearch.aget_by_ids(nonexistent_ids)
    assert len(docs) == 0
