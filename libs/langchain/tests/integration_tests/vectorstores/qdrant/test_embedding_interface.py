import uuid
from typing import Callable, Optional

import pytest

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Qdrant
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)


@pytest.mark.parametrize(
    ["embeddings", "embedding_function"],
    [
        (ConsistentFakeEmbeddings(), None),
        (ConsistentFakeEmbeddings().embed_query, None),
        (None, ConsistentFakeEmbeddings().embed_query),
    ],
)
def test_qdrant_embedding_interface(
    embeddings: Optional[Embeddings], embedding_function: Optional[Callable]
) -> None:
    """Test Qdrant may accept different types for embeddings."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    collection_name = uuid.uuid4().hex

    Qdrant(
        client,
        collection_name,
        embeddings=embeddings,
        embedding_function=embedding_function,
    )


@pytest.mark.parametrize(
    ["embeddings", "embedding_function"],
    [
        (ConsistentFakeEmbeddings(), ConsistentFakeEmbeddings().embed_query),
        (None, None),
    ],
)
def test_qdrant_embedding_interface_raises_value_error(
    embeddings: Optional[Embeddings], embedding_function: Optional[Callable]
) -> None:
    """Test Qdrant requires only one method for embeddings."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    collection_name = uuid.uuid4().hex

    with pytest.raises(ValueError):
        Qdrant(
            client,
            collection_name,
            embeddings=embeddings,
            embedding_function=embedding_function,
        )
