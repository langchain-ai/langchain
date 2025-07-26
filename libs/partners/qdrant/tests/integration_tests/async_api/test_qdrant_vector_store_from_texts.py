import uuid

import pytest
from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    ConsistentFakeSparseEmbeddings,
    assert_documents_equals,
)
from tests.integration_tests.fixtures import qdrant_locations, retrieval_modes


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
async def test_async_vectorstore_from_texts(
    location: str, retrieval_mode: RetrievalMode
) -> None:
    """Test end to end QdrantVectorStore async construction from texts."""
    collection_name = uuid.uuid4().hex

    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        collection_name=collection_name,
        client_options={"location": location},
    )

    # Add texts using async method
    await vec_store.aadd_texts(["Lorem ipsum dolor sit amet", "Ipsum dolor sit amet"])

    # Verify count using AsyncQdrantClient
    async_client = vec_store.client
    assert isinstance(async_client, AsyncQdrantClient)
    count_result = await async_client.count(collection_name)
    assert 2 == count_result.count


@pytest.mark.parametrize("location", qdrant_locations())
async def test_async_qdrant_similarity_search(location: str) -> None:
    """Test QdrantVectorStore async similarity search."""
    collection_name = uuid.uuid4().hex

    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        client_options={"location": location},
    )

    await vec_store.aadd_texts(["foo", "bar", "baz"])

    # Test async similarity search
    output = await vec_store.asimilarity_search("foo", k=1)
    assert len(output) == 1
    # Use assert_documents_equals which doesn't assume ordering
    assert_documents_equals(actual=output, expected=[Document(page_content="foo")])


@pytest.mark.parametrize("location", qdrant_locations())
async def test_async_qdrant_delete(location: str) -> None:
    """Test QdrantVectorStore async delete functionality."""
    collection_name = uuid.uuid4().hex
    texts = ["foo", "bar", "baz"]
    ids = [
        "fa38d572-4c31-4579-aedc-1960d79df6df",
        "cdc1aa36-d6ab-4fb2-8a94-56674fd27484",
        "b4c1aa36-d6ab-4fb2-8a94-56674fd27485",
    ]

    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        client_options={"location": location},
    )

    await vec_store.aadd_texts(texts, ids=ids)

    async_client = vec_store.client
    assert isinstance(async_client, AsyncQdrantClient)

    # Verify all texts are added
    count_result = await async_client.count(collection_name)
    assert 3 == count_result.count

    # Delete one document
    result = await vec_store.adelete([ids[1]])  # Delete the second document
    assert result is True

    # Verify deletion
    count_result = await async_client.count(collection_name)
    assert 2 == count_result.count


@pytest.mark.parametrize("location", qdrant_locations())
async def test_async_qdrant_add_documents(location: str) -> None:
    """Test QdrantVectorStore async add documents functionality."""
    collection_name = uuid.uuid4().hex

    documents = [
        Document(page_content="foo", metadata={"page": 1}),
        Document(page_content="bar", metadata={"page": 2}),
        Document(page_content="baz", metadata={"page": 3}),
    ]

    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        client_options={"location": location},
    )

    # Test async add documents
    ids = await vec_store.aadd_documents(documents)
    assert len(ids) == 3
    assert all(isinstance(id_, str) for id_ in ids)

    async_client = vec_store.client
    assert isinstance(async_client, AsyncQdrantClient)

    # Verify documents are added
    count_result = await async_client.count(collection_name)
    assert 3 == count_result.count
