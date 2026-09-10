from __future__ import annotations

import uuid

import pytest
from langchain_core.documents import Document

from langchain_qdrant import QdrantVectorStore
from tests.integration_tests.common import ConsistentFakeEmbeddings
from tests.integration_tests.fixtures import qdrant_locations


@pytest.mark.parametrize("location", qdrant_locations())
async def test_qdrant_vector_store_aadd_documents_with_custom_ids(
    location: str,
) -> None:
    """Regression test for #32283.

    `aadd_documents` with custom ids must not raise
    `TypeError: aadd_texts() got multiple values for keyword argument 'ids'`.
    """
    docsearch = QdrantVectorStore.from_texts(
        ["foo"],
        ConsistentFakeEmbeddings(),
        location=location,
    )
    documents = [
        Document(page_content="Hello world"),
        Document(page_content="Goodbye world"),
    ]
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    returned_ids = await docsearch.aadd_documents(documents, ids=ids)

    assert returned_ids == ids
    output = docsearch.similarity_search("Hello world", k=1)
    assert output[0].page_content == "Hello world"
