"""Async add texts tests for QdrantVectorStore."""

from __future__ import annotations

import uuid

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    ConsistentFakeSparseEmbeddings,
    assert_documents_equals,
)
from tests.integration_tests.fixtures import qdrant_locations, retrieval_modes


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
async def test_async_add_documents_extends_existing_collection(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
) -> None:
    """Test async add_documents extends existing collection."""
    texts = ["foo", "bar", "baz"]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    new_texts = ["foobar", "foobaz"]
    await docsearch.aadd_documents(
        [Document(page_content=content) for content in new_texts]
    )
    output = await docsearch.asimilarity_search("foobar", k=1)
    assert_documents_equals(output, [Document(page_content="foobar")])


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
@pytest.mark.parametrize("batch_size", [1, 64])
async def test_async_add_texts_returns_all_ids(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
    batch_size: int,
) -> None:
    """Test async add_texts returns unique ids."""
    docsearch = QdrantVectorStore.from_texts(
        ["foobar"],
        ConsistentFakeEmbeddings(),
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        batch_size=batch_size,
    )

    ids = await docsearch.aadd_texts(["foo", "bar", "baz"])
    assert len(ids) == 3
    assert len(set(ids)) == 3
    # Use async get_by_ids
    docs = await docsearch.aget_by_ids(ids)
    assert len(docs) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
async def test_async_add_texts_stores_duplicated_texts(
    location: str,
    vector_name: str,
) -> None:
    """Test async add_texts stores duplicated texts separately."""
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
    ids = await vec_store.aadd_texts(["abc", "abc"], [{"a": 1}, {"a": 2}])

    assert len(set(ids)) == 2
    assert client.count(collection_name).count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
@pytest.mark.parametrize("batch_size", [1, 64])
async def test_async_add_texts_stores_ids(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
    batch_size: int,
) -> None:
    """Test async add_texts stores provided ids."""
    ids: list[str | int] = [
        "fa38d572-4c31-4579-aedc-1960d79df6df",
        432,
        432145435,
    ]
    collection_name = uuid.uuid4().hex
    vec_store = QdrantVectorStore.from_texts(
        ["abc", "def", "ghi"],
        ConsistentFakeEmbeddings(),
        ids=ids,
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        batch_size=batch_size,
    )

    assert vec_store.client.count(collection_name).count == 3
    stored_ids = [point.id for point in vec_store.client.scroll(collection_name)[0]]
    assert set(ids) == set(stored_ids)
    # Use async get_by_ids
    docs = await vec_store.aget_by_ids(ids)
    assert len(docs) == 3
