import uuid

import pytest
from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, models

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    ConsistentFakeSparseEmbeddings,
)
from tests.integration_tests.fixtures import qdrant_locations, retrieval_modes


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
async def test_async_add_texts_basic(location: str, retrieval_mode: RetrievalMode) -> None:
    """Test async basic add_texts functionality."""
    collection_name = uuid.uuid4().hex
    
    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        collection_name=collection_name,
        client_options={"location": location},
    )

    # Add initial texts
    texts1 = ["foo", "bar"]
    ids1 = await vec_store.aadd_texts(texts1)
    assert len(ids1) == 2

    # Add more texts
    texts2 = ["baz", "qux"]
    ids2 = await vec_store.aadd_texts(texts2)
    assert len(ids2) == 2

    # Verify all texts are in the collection
    async_client = vec_store.client
    assert isinstance(async_client, AsyncQdrantClient)
    count_result = await async_client.count(collection_name)
    assert 4 == count_result.count

    # Test search functionality
    results = await vec_store.asimilarity_search("foo", k=1)
    assert len(results) == 1
    assert results[0].page_content == "foo"


@pytest.mark.parametrize("location", qdrant_locations())
async def test_async_add_texts_with_filters(location: str) -> None:
    """Test async add_texts and search with filters."""
    collection_name = uuid.uuid4().hex
    
    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        client_options={"location": location},
    )

    texts = ["Red apple", "Blue apple", "Green apple"]
    metadatas = [
        {"color": "red", "type": "fruit"},
        {"color": "blue", "type": "fruit"},
        {"color": "green", "type": "fruit"},
    ]
    
    await vec_store.aadd_texts(texts, metadatas=metadatas)

    # Test search with filter
    filter_condition = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.color",
                match=models.MatchValue(value="red")
            )
        ]
    )
    
    results = await vec_store.asimilarity_search(
        "apple", k=3, filter=filter_condition
    )
    
    assert len(results) == 1
    assert results[0].page_content == "Red apple"
    assert results[0].metadata["color"] == "red"


@pytest.mark.parametrize("location", qdrant_locations())
async def test_async_add_texts_with_custom_ids(location: str) -> None:
    """Test async add_texts with custom IDs."""
    collection_name = uuid.uuid4().hex
    
    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        client_options={"location": location},
    )

    texts = ["First document", "Second document"]
    custom_ids = [
        "fa38d572-4c31-4579-aedc-1960d79df6df",
        "cdc1aa36-d6ab-4fb2-8a94-56674fd27484"
    ]
    
    returned_ids = await vec_store.aadd_texts(texts, ids=custom_ids)
    
    # Should return the same IDs we provided
    assert returned_ids == custom_ids

    # Verify documents can be retrieved by custom IDs
    docs = await vec_store.aget_by_ids(custom_ids)
    assert len(docs) == 2
    
    contents = [doc.page_content for doc in docs]
    assert "First document" in contents
    assert "Second document" in contents
