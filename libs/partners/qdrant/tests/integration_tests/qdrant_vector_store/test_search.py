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


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize("batch_size", [1, 64])
def test_similarity_search(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    batch_size: int,
) -> None:
    """Test end to end construction and search."""
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
    output = docsearch.similarity_search("foo", k=1)
    assert_documents_equals(actual=output, expected=[Document(page_content="foo")])


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("content_payload_key", [QdrantVectorStore.CONTENT_KEY, "foo"])
@pytest.mark.parametrize(
    "metadata_payload_key", [QdrantVectorStore.METADATA_KEY, "bar"]
)
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_similarity_search_by_vector(
    location: str,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: str,
    batch_size: int,
) -> None:
    """Test end to end construction and search."""
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
    output = docsearch.similarity_search_by_vector(embeddings, k=1)
    assert_documents_equals(output, [Document(page_content="foo")])


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("content_payload_key", [QdrantVectorStore.CONTENT_KEY, "foo"])
@pytest.mark.parametrize(
    "metadata_payload_key", [QdrantVectorStore.METADATA_KEY, "bar"]
)
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_similarity_search_with_score_by_vector(
    location: str,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: str,
    batch_size: int,
) -> None:
    """Test end to end construction and search."""
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
    output = docsearch.similarity_search_with_score_by_vector(embeddings, k=1)
    assert len(output) == 1
    document, score = output[0]
    assert_documents_equals([document], [Document(page_content="foo")])
    assert score >= 0


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize(
    "metadata_payload_key", [QdrantVectorStore.METADATA_KEY, "bar"]
)
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
def test_similarity_search_filters(
    location: str,
    metadata_payload_key: str,
    retrieval_mode: RetrievalMode,
) -> None:
    """Test end to end construction and search."""
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
    output = docsearch.similarity_search("foo", k=1, filter=qdrant_filter)

    assert_documents_equals(
        actual=output,
        expected=[
            Document(
                page_content="bar",
                metadata={"page": 1, "metadata": {"page": 2, "pages": [3, -1]}},
            )
        ],
    )


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
def test_similarity_relevance_search_no_threshold(
    location: str,
    vector_name: str,
) -> None:
    """Test end to end construction and search."""
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
    output = docsearch.similarity_search_with_relevance_scores(
        "foo", k=3, score_threshold=None
    )
    assert len(output) == 3
    for i in range(len(output)):
        assert round(output[i][1], 2) >= 0
        assert round(output[i][1], 2) <= 1


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
def test_relevance_search_with_threshold(
    location: str,
    vector_name: str,
) -> None:
    """Test end to end construction and search."""
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
    kwargs = {"score_threshold": score_threshold}
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3, **kwargs)
    assert len(output) == 1
    assert all([score >= score_threshold for _, score in output])


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("content_payload_key", [QdrantVectorStore.CONTENT_KEY, "foo"])
@pytest.mark.parametrize(
    "metadata_payload_key", [QdrantVectorStore.METADATA_KEY, "bar"]
)
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
def test_relevance_search_with_threshold_and_filter(
    location: str,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: str,
) -> None:
    """Test end to end construction and search."""
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
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        vector_name=vector_name,
    )
    score_threshold = 0.99  # for almost exact match
    negative_filter = models.Filter(
        must=[
            models.FieldCondition(
                key=f"{metadata_payload_key}.page", match=models.MatchValue(value=1)
            )
        ]
    )
    kwargs = {"filter": negative_filter, "score_threshold": score_threshold}
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3, **kwargs)
    assert len(output) == 0
    positive_filter = models.Filter(
        must=[
            models.FieldCondition(
                key=f"{metadata_payload_key}.page", match=models.MatchValue(value=0)
            )
        ]
    )
    kwargs = {"filter": positive_filter, "score_threshold": score_threshold}
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3, **kwargs)
    assert len(output) == 1
    assert all([score >= score_threshold for _, score in output])


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("content_payload_key", [QdrantVectorStore.CONTENT_KEY, "foo"])
@pytest.mark.parametrize(
    "metadata_payload_key", [QdrantVectorStore.METADATA_KEY, "bar"]
)
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
def test_similarity_search_filters_with_qdrant_filters(
    location: str,
    content_payload_key: str,
    metadata_payload_key: str,
    retrieval_mode: RetrievalMode,
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "details": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=location,
        metadatas=metadatas,
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    qdrant_filter = models.Filter(
        must=[
            models.FieldCondition(
                key=content_payload_key, match=models.MatchValue(value="bar")
            ),
            models.FieldCondition(
                key=f"{metadata_payload_key}.page",
                match=models.MatchValue(value=1),
            ),
            models.FieldCondition(
                key=f"{metadata_payload_key}.details.page",
                match=models.MatchValue(value=2),
            ),
            models.FieldCondition(
                key=f"{metadata_payload_key}.details.pages",
                match=models.MatchAny(any=[3]),
            ),
        ]
    )
    output = docsearch.similarity_search("foo", k=1, filter=qdrant_filter)
    assert_documents_equals(
        actual=output,
        expected=[
            Document(
                page_content="bar",
                metadata={"page": 1, "details": {"page": 2, "pages": [3, -1]}},
            )
        ],
    )
