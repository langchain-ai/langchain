"""Async search tests for QdrantVectorStore."""

import pytest
from langchain_core.documents import Document
from qdrant_client import models

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
@pytest.mark.parametrize("batch_size", [1, 64])
async def test_async_similarity_search(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    batch_size: int,
) -> None:
    """Test async similarity search."""
    texts = ["foo", "bar", "baz"]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=location,
        batch_size=batch_size,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert_documents_equals(actual=output, expected=[Document(page_content="foo")])


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("content_payload_key", [QdrantVectorStore.CONTENT_KEY, "foo"])
@pytest.mark.parametrize(
    "metadata_payload_key", [QdrantVectorStore.METADATA_KEY, "bar"]
)
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("batch_size", [1, 64])
async def test_async_similarity_search_by_vector(
    location: str,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: str,
    batch_size: int,
) -> None:
    """Test async similarity search by vector."""
    texts = ["foo", "bar", "baz"]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=location,
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
        vector_name=vector_name,
    )
    embeddings = ConsistentFakeEmbeddings().embed_query("foo")
    output = await docsearch.asimilarity_search_by_vector(embeddings, k=1)
    assert_documents_equals(output, [Document(page_content="foo")])


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("content_payload_key", [QdrantVectorStore.CONTENT_KEY, "foo"])
@pytest.mark.parametrize(
    "metadata_payload_key", [QdrantVectorStore.METADATA_KEY, "bar"]
)
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("batch_size", [1, 64])
async def test_async_similarity_search_with_score_by_vector(
    location: str,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: str,
    batch_size: int,
) -> None:
    """Test async similarity search with score by vector."""
    texts = ["foo", "bar", "baz"]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=location,
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
        vector_name=vector_name,
    )
    embeddings = ConsistentFakeEmbeddings().embed_query("foo")
    output = await docsearch.asimilarity_search_with_score_by_vector(embeddings, k=1)
    assert len(output) == 1
    document, score = output[0]
    assert_documents_equals([document], [Document(page_content="foo")])
    assert score >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize(
    "metadata_payload_key", [QdrantVectorStore.METADATA_KEY, "bar"]
)
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
async def test_async_similarity_search_filters(
    location: str,
    metadata_payload_key: str,
    retrieval_mode: RetrievalMode,
) -> None:
    """Test async similarity search with filters."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "metadata": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        location=location,
        metadata_payload_key=metadata_payload_key,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    qdrant_filter = models.Filter(
        must=[
            models.FieldCondition(
                key=f"{metadata_payload_key}.page", match=models.MatchValue(value=1)
            )
        ]
    )
    output = await docsearch.asimilarity_search("foo", k=1, filter=qdrant_filter)

    assert_documents_equals(
        actual=output,
        expected=[
            Document(
                page_content="bar",
                metadata={"page": 1, "metadata": {"page": 2, "pages": [3, -1]}},
            )
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
async def test_async_relevance_search_no_threshold(
    location: str,
    vector_name: str,
) -> None:
    """Test async relevance search without threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "metadata": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        location=location,
        vector_name=vector_name,
    )
    output = await docsearch.asimilarity_search_with_score("foo", k=3)
    assert len(output) == 3
    for _, score in output:
        assert score >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
async def test_async_relevance_search_with_threshold(
    location: str,
    vector_name: str,
) -> None:
    """Test async relevance search with score threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "metadata": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        location=location,
        vector_name=vector_name,
    )

    score_threshold = 0.99
    output = await docsearch.asimilarity_search_with_score(
        "foo", k=3, score_threshold=score_threshold
    )
    assert len(output) == 1
    assert all(score >= score_threshold for _, score in output)
