from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest  # type: ignore[import-not-found]

from langchain_qdrant import Qdrant
from tests.integration_tests.common import ConsistentFakeEmbeddings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


@pytest.mark.parametrize(
    ("embeddings", "embedding_function"),
    [
        (ConsistentFakeEmbeddings(), None),
        (ConsistentFakeEmbeddings().embed_query, None),
        (None, ConsistentFakeEmbeddings().embed_query),
    ],
)
def test_qdrant_embedding_interface(
    embeddings: Embeddings | None, embedding_function: Callable | None
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
    ("embeddings", "embedding_function"),
    [
        (ConsistentFakeEmbeddings(), ConsistentFakeEmbeddings().embed_query),
        (None, None),
    ],
)
def test_qdrant_embedding_interface_raises_value_error(
    embeddings: Embeddings | None, embedding_function: Callable | None
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
