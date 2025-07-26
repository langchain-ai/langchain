import uuid

import pytest
from langchain_core.documents import Document
from qdrant_client import models

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    ConsistentFakeSparseEmbeddings,
)
from tests.integration_tests.fixtures import qdrant_locations, retrieval_modes


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
async def test_async_similarity_search_basic(
    location: str, retrieval_mode: RetrievalMode
) -> None:
    """Test basic async similarity search functionality."""
    collection_name = uuid.uuid4().hex

    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        collection_name=collection_name,
        client_options={"location": location},
    )

    texts = ["apple", "banana", "cherry", "date"]
    await vec_store.aadd_texts(texts)

    # Test basic similarity search
    results = await vec_store.asimilarity_search("apple", k=2)

    assert len(results) <= 2
    assert all(isinstance(doc, Document) for doc in results)
    assert results[0].page_content == "apple"  # Should be most similar to itself


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
async def test_async_similarity_search_with_score(
    location: str, retrieval_mode: RetrievalMode
) -> None:
    """Test async similarity search with scores."""
    collection_name = uuid.uuid4().hex

    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        collection_name=collection_name,
        client_options={"location": location},
    )

    texts = ["red apple", "green apple", "blue car", "yellow banana"]
    await vec_store.aadd_texts(texts)

    # Test similarity search with scores
    results = await vec_store.asimilarity_search_with_score("apple", k=3)

    assert len(results) <= 3
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert score >= 0.0  # Scores should be non-negative

    # First result should be most relevant
    all_contents = [doc.page_content for doc, _ in results]
    assert any("apple" in content for content in all_contents)


@pytest.mark.parametrize("location", qdrant_locations())
async def test_async_similarity_search_empty_collection(location: str) -> None:
    """Test async similarity search on empty collection."""
    collection_name = uuid.uuid4().hex

    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        client_options={"location": location},
    )

    # Search in empty collection
    results = await vec_store.asimilarity_search("anything", k=5)

    assert len(results) == 0


@pytest.mark.parametrize("location", qdrant_locations())
async def test_async_similarity_search_with_consistency(location: str) -> None:
    """Test async similarity search with read consistency parameter."""
    collection_name = uuid.uuid4().hex

    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        client_options={"location": location},
    )

    texts = ["test document"]
    await vec_store.aadd_texts(texts)

    # Test with different consistency levels
    consistency_levels = [
        1,
        models.ReadConsistencyType.MAJORITY,
        models.ReadConsistencyType.ALL,
    ]

    for consistency in consistency_levels:
        results = await vec_store.asimilarity_search(
            "test", k=1, consistency=consistency
        )

        assert len(results) <= 1
        if results:
            assert results[0].page_content == "test document"
