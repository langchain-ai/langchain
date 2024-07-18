import uuid

import pytest

from langchain_qdrant.qdrant import QdrantVectorStore, RetrievalMode
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    ConsistentFakeSparseEmbeddings,
)
from tests.integration_tests.fixtures import qdrant_locations, retrieval_modes


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
def test_qdrant_from_existing_collection_uses_same_collection(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
) -> None:
    """Test if the QdrantVectorStore.from_existing_collection reuses the collection."""

    collection_name = uuid.uuid4().hex
    docs = ["foo"]
    QdrantVectorStore.from_texts(
        docs,
        embedding=ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    qdrant = QdrantVectorStore.from_existing_collection(
        collection_name,
        embedding=ConsistentFakeEmbeddings(),
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )
    qdrant.add_texts(["baz", "bar"])

    assert 3 == qdrant.client.count(collection_name).count
