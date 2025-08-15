from __future__ import annotations

from typing import Optional

import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document

from langchain_qdrant import Qdrant
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    assert_documents_equals,
)


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "test_content"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "test_metadata"])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
def test_qdrant_max_marginal_relevance_search(
    batch_size: int,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and MRR search."""
    from qdrant_client import models

    filter_ = models.Filter(
        must=[
            models.FieldCondition(
                key=f"{metadata_payload_key}.page",
                match=models.MatchValue(
                    value=2,
                ),
            ),
        ],
    )

    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        location=":memory:",
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
        vector_name=vector_name,
        distance_func="EUCLID",  # Euclid distance used to avoid normalization
    )
    output = docsearch.max_marginal_relevance_search(
        "foo", k=2, fetch_k=3, lambda_mult=0.0
    )
    assert_documents_equals(
        output,
        [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="baz", metadata={"page": 2}),
        ],
    )

    output = docsearch.max_marginal_relevance_search(
        "foo", k=2, fetch_k=3, lambda_mult=0.0, filter=filter_
    )
    assert_documents_equals(
        output,
        [Document(page_content="baz", metadata={"page": 2})],
    )
