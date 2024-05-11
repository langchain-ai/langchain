from typing import Optional

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores import Qdrant
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)
from tests.integration_tests.vectorstores.qdrant.common import assert_documents_equals


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
def test_qdrant_chunk_window_search(
    batch_size: int,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and search."""
    texts = [
        "Dogs are known for their loyalty and companionship",
        "Dogs have temperaments ranging from energetic and playful to calm and gentle",
        "Dogs are highly intelligent animals",
        "Cars come in various shapes, sizes, and colors",
        "Drivers often customize cars to reflect their personality",
        "Cars enable mobility, facilitating travel and exploration",
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        ids=[1, 2, 3, 4, 5, 6],
        location=":memory:",
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
        vector_name=vector_name,
    )
    output = docsearch.chunk_window_retrieval_similarity_search(
        "Dogs have temperaments ranging from energetic and playful to calm and gentle",
        k=1,
        window_size=1,
    )
    assert_documents_equals(
        actual=output,
        expected=[
            Document(page_content="Dogs are known for their loyalty and companionship"),
            Document(
                page_content="Dogs have temperaments ranging from energetic \
                  and playful to calm and gentle"
            ),
            Document(page_content="Dogs are highly intelligent animals"),
        ],
    )

    # below assert highlights that even though the chunks are not semantically
    # similar, they will still be returned based on window_size
    # we are not reranking the chunks in this approach.
    output = docsearch.chunk_window_retrieval_similarity_search(
        "Dogs are highly intelligent animals", k=1, window_size=1
    )
    assert_documents_equals(
        actual=output,
        expected=[
            Document(
                page_content="Dogs have temperaments ranging from energetic \
                and playful to calm and gentle"
            ),
            Document(page_content="Dogs are highly intelligent animals"),
            Document(page_content="Cars come in various shapes, sizes, and colors"),
        ],
    )


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
def test_qdrant_chunk_window_search_by_vector(
    batch_size: int,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and search."""
    texts = [
        "Dogs are known for their loyalty and companionship",
        "Dogs have temperaments ranging from energetic and playful to calm and gentle",
        "Dogs are highly intelligent animals",
        "Cars come in various shapes, sizes, and colors",
        "Drivers often customize cars to reflect their personality",
        "Cars enable mobility, facilitating travel and exploration",
    ]
    embeddings = ConsistentFakeEmbeddings()
    docsearch = Qdrant.from_texts(
        texts,
        embeddings,
        ids=[1, 2, 3, 4, 5, 6],
        location=":memory:",
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
        vector_name=vector_name,
    )
    embed_query = embeddings.embed_query(
        "Dogs have temperaments ranging from energetic and playful to calm and gentle"
    )
    output = docsearch.chunk_window_retrieval_similarity_search_by_vector(
        embed_query, k=1, window_size=1
    )
    assert_documents_equals(
        actual=output,
        expected=[
            Document(page_content="Dogs are known for their loyalty and companionship"),
            Document(
                page_content="Dogs have temperaments ranging from energetic \
                  and playful to calm and gentle"
            ),
            Document(page_content="Dogs are highly intelligent animals"),
        ],
    )

    # below assert highlights that even though the chunks are not semantically
    # similar, they will still be returned based on window_size
    # we are not reranking the chunks in this approach.
    embed_query = embeddings.embed_query("Dogs are highly intelligent animals")
    output = docsearch.chunk_window_retrieval_similarity_search_by_vector(
        embed_query, k=1, window_size=1
    )
    assert_documents_equals(
        actual=output,
        expected=[
            Document(
                page_content="Dogs have temperaments ranging from energetic \
                  and playful to calm and gentle"
            ),
            Document(page_content="Dogs are highly intelligent animals"),
            Document(page_content="Cars come in various shapes, sizes, and colors"),
        ],
    )


@pytest.mark.parametrize("vector_name", [None, "my-vector"])
def test_qdrant_chunk_window_search_filters_with_qdrant_filters(
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and search."""
    from qdrant_client.http import models as rest

    texts = [
        "Dogs are known for their loyalty and companionship",
        "Dogs have temperaments ranging from energetic and playful to calm and gentle",
        "Dogs are highly intelligent animals",
        "Cars come in various shapes, sizes, and colors",
        "Drivers often customize cars to reflect their personality",
        "Cars enable mobility, facilitating travel and exploration",
    ]
    metadatas = [
        {"page": i, "details": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        ids=[1, 2, 3, 4, 5, 6],
        location=":memory:",
        vector_name=vector_name,
    )

    qdrant_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="metadata.page",
                match=rest.MatchValue(value=2),
            ),
            rest.FieldCondition(
                key="metadata.details.page",
                match=rest.MatchValue(value=3),
            ),
            rest.FieldCondition(
                key="metadata.details.pages",
                match=rest.MatchAny(any=[4]),
            ),
        ]
    )
    output = docsearch.chunk_window_retrieval_similarity_search(
        "Dogs have temperaments ranging from energetic and playful to calm and gentle",
        k=1,
        window_size=1,
        filter=qdrant_filter,
    )
    assert_documents_equals(
        actual=output,
        expected=[
            Document(
                page_content="Dogs have temperaments ranging from energetic \
                  and playful to calm and gentle",
                metadata={"page": 1, "details": {"page": 2, "pages": [3, -1]}},
            ),
            Document(
                page_content="Dogs are highly intelligent animals",
                metadata={"page": 2, "details": {"page": 3, "pages": [4, -1]}},
            ),
            Document(
                page_content="Cars come in various shapes, sizes, and colors",
                metadata={"page": 3, "details": {"page": 4, "pages": [5, -1]}},
            ),
        ],
    )
