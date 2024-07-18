import tempfile
import uuid

import pytest

from langchain_community.vectorstores import Qdrant
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)


@pytest.mark.parametrize("vector_name", ["custom-vector"])
def test_qdrant_from_existing_collection_uses_same_collection(vector_name: str) -> None:
    """Test if the Qdrant.from_existing_collection reuses the same collection."""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmpdir:
        docs = ["foo"]
        qdrant = Qdrant.from_texts(
            docs,
            embedding=ConsistentFakeEmbeddings(),
            path=str(tmpdir),
            collection_name=collection_name,
            vector_name=vector_name,
        )
        del qdrant

        qdrant = Qdrant.from_existing_collection(
            embedding=ConsistentFakeEmbeddings(),
            path=str(tmpdir),
            collection_name=collection_name,
            vector_name=vector_name,
        )
        qdrant.add_texts(["baz", "bar"])
        del qdrant

        client = QdrantClient(path=str(tmpdir))
        assert 3 == client.count(collection_name).count
