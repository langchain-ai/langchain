import uuid

import pytest
from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, models

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from tests.integration_tests.common import ConsistentFakeEmbeddings
from tests.integration_tests.fixtures import qdrant_locations


@pytest.mark.parametrize("location", qdrant_locations())
async def test_async_max_marginal_relevance_search_basic(location: str) -> None:
    """Test basic async max marginal relevance search functionality."""
    collection_name = uuid.uuid4().hex
    
    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        retrieval_mode=RetrievalMode.DENSE,  # MMR only works with dense
        collection_name=collection_name,
        client_options={"location": location},
    )

    texts = ["apple", "banana", "cherry", "apple pie", "apple juice"]
    await vec_store.aadd_texts(texts)

    # Test basic MMR search
    results = await vec_store.amax_marginal_relevance_search(
        "apple", k=3, fetch_k=5
    )
    
    assert len(results) <= 3
    assert all(isinstance(doc, Document) for doc in results)
    
    # First result should be most similar
    assert "apple" in results[0].page_content.lower()


@pytest.mark.parametrize("location", qdrant_locations())
async def test_async_max_marginal_relevance_search_by_vector(location: str) -> None:
    """Test async MMR search by vector."""
    collection_name = uuid.uuid4().hex
    
    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        retrieval_mode=RetrievalMode.DENSE,
        collection_name=collection_name,
        client_options={"location": location},
    )

    texts = ["apple", "banana", "cherry", "apple pie"]
    await vec_store.aadd_texts(texts)

    # Get embedding for search
    embedding = ConsistentFakeEmbeddings().embed_query("apple")
    
    # Test MMR by vector
    results = await vec_store.amax_marginal_relevance_search_by_vector(
        embedding, k=2, fetch_k=4
    )
    
    assert len(results) <= 2
    assert all(isinstance(doc, Document) for doc in results)


@pytest.mark.parametrize("location", qdrant_locations())
async def test_async_max_marginal_relevance_search_with_score_by_vector(location: str) -> None:
    """Test async MMR search with score by vector."""
    collection_name = uuid.uuid4().hex
    
    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        retrieval_mode=RetrievalMode.DENSE,
        collection_name=collection_name,
        client_options={"location": location},
    )

    texts = ["apple", "banana", "cherry", "apple pie", "apple juice"]
    await vec_store.aadd_texts(texts)

    # Get embedding for search
    embedding = ConsistentFakeEmbeddings().embed_query("apple")
    
    # Test MMR with scores by vector
    results = await vec_store.amax_marginal_relevance_search_with_score_by_vector(
        embedding, k=3, fetch_k=5
    )
    
    assert len(results) <= 3
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert score >= 0.0


@pytest.mark.parametrize("location", qdrant_locations())
async def test_async_max_marginal_relevance_search_empty_collection(location: str) -> None:
    """Test async MMR search on empty collection."""
    collection_name = uuid.uuid4().hex
    
    vec_store = await QdrantVectorStore.aconstruct_instance(
        embedding=ConsistentFakeEmbeddings(),
        retrieval_mode=RetrievalMode.DENSE,
        collection_name=collection_name,
        client_options={"location": location},
    )

    # Search in empty collection
    results = await vec_store.amax_marginal_relevance_search(
        "anything", k=5, fetch_k=10
    )
    
    assert len(results) == 0
