from typing import Optional

import numpy as np
import pytest
from qdrant_client.http import models as rest

from langchain.schema import Document
from langchain.vectorstores import Qdrant
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)

from .common import qdrant_is_not_running

# Skipping all the tests in the module if Qdrant is not running on localhost.
pytestmark = pytest.mark.skipif(
    qdrant_is_not_running(), reason="Qdrant server is not running"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
async def test_qdrant_similarity_search(
    batch_size: int,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
        vector_name=vector_name,
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
async def test_qdrant_similarity_search_by_vector(
    batch_size: int,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
        vector_name=vector_name,
    )
    embeddings = ConsistentFakeEmbeddings().embed_query("foo")
    output = await docsearch.asimilarity_search_by_vector(embeddings, k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
async def test_qdrant_similarity_search_with_score_by_vector(
    batch_size: int,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
        vector_name=vector_name,
    )
    embeddings = ConsistentFakeEmbeddings().embed_query("foo")
    output = await docsearch.asimilarity_search_with_score_by_vector(embeddings, k=1)
    assert len(output) == 1
    document, score = output[0]
    assert document == Document(page_content="foo")
    assert score >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
async def test_qdrant_similarity_search_filters(
    batch_size: int, vector_name: Optional[str]
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "metadata": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        batch_size=batch_size,
        vector_name=vector_name,
    )

    output = await docsearch.asimilarity_search(
        "foo", k=1, filter={"page": 1, "metadata": {"page": 2, "pages": [3]}}
    )
    assert output == [
        Document(
            page_content="bar",
            metadata={"page": 1, "metadata": {"page": 2, "pages": [3, -1]}},
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
async def test_qdrant_similarity_search_with_relevance_score_no_threshold(
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "metadata": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        vector_name=vector_name,
    )
    output = await docsearch.asimilarity_search_with_relevance_scores(
        "foo", k=3, score_threshold=None
    )
    assert len(output) == 3
    for i in range(len(output)):
        assert round(output[i][1], 2) >= 0
        assert round(output[i][1], 2) <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
async def test_qdrant_similarity_search_with_relevance_score_with_threshold(
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "metadata": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        vector_name=vector_name,
    )

    score_threshold = 0.98
    kwargs = {"score_threshold": score_threshold}
    output = await docsearch.asimilarity_search_with_relevance_scores(
        "foo", k=3, **kwargs
    )
    assert len(output) == 1
    assert all([score >= score_threshold for _, score in output])


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
async def test_similarity_search_with_relevance_score_with_threshold_and_filter(
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "metadata": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        vector_name=vector_name,
    )
    score_threshold = 0.99  # for almost exact match
    # test negative filter condition
    negative_filter = {"page": 1, "metadata": {"page": 2, "pages": [3]}}
    kwargs = {"filter": negative_filter, "score_threshold": score_threshold}
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3, **kwargs)
    assert len(output) == 0
    # test positive filter condition
    positive_filter = {"page": 0, "metadata": {"page": 1, "pages": [2]}}
    kwargs = {"filter": positive_filter, "score_threshold": score_threshold}
    output = await docsearch.asimilarity_search_with_relevance_scores(
        "foo", k=3, **kwargs
    )
    assert len(output) == 1
    assert all([score >= score_threshold for _, score in output])


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
async def test_qdrant_similarity_search_filters_with_qdrant_filters(
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "details": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        vector_name=vector_name,
    )

    qdrant_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="metadata.page",
                match=rest.MatchValue(value=1),
            ),
            rest.FieldCondition(
                key="metadata.details.page",
                match=rest.MatchValue(value=2),
            ),
            rest.FieldCondition(
                key="metadata.details.pages",
                match=rest.MatchAny(any=[3]),
            ),
        ]
    )
    output = await docsearch.asimilarity_search("foo", k=1, filter=qdrant_filter)
    assert output == [
        Document(
            page_content="bar",
            metadata={"page": 1, "details": {"page": 2, "pages": [3, -1]}},
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
async def test_qdrant_similarity_search_with_relevance_scores(
    batch_size: int,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: str,
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
        vector_name=vector_name,
    )
    output = await docsearch.asimilarity_search_with_relevance_scores("foo", k=3)

    assert all(
        (1 >= score or np.isclose(score, 1)) and score >= 0 for _, score in output
    )
