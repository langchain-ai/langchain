"""Regression tests for generator input to ``add_texts`` (issue #41790).

``texts`` is typed as ``Iterable[str]``, but when ``ids`` was omitted the batch
helpers built default ids up front by iterating ``texts``, which exhausted a
generator before the batching loop ran: nothing was stored and ``[]`` was
returned without any error.
"""

from __future__ import annotations

import uuid

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from langchain_qdrant import Qdrant, QdrantVectorStore

TEXTS = ["alpha", "beta", "gamma"]


class FakeEmbeddings(Embeddings):
    """Deterministic two-dimensional embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(i)] for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _create_collection(client: QdrantClient, collection_name: str) -> None:
    client.create_collection(
        collection_name,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )


def test_qdrant_vector_store_add_texts_generator_stores_all_texts() -> None:
    """A generator of texts must be fully stored, not silently dropped."""
    client = QdrantClient(":memory:")
    _create_collection(client, "c")
    store = QdrantVectorStore(
        client=client, collection_name="c", embedding=FakeEmbeddings()
    )

    ids = store.add_texts(text for text in TEXTS)

    assert len(ids) == 3
    assert client.count("c").count == 3


def test_qdrant_vector_store_add_texts_generator_stores_all_texts_in_batches() -> None:
    """Generator input must survive batching and return every id."""
    client = QdrantClient(":memory:")
    _create_collection(client, "c")
    store = QdrantVectorStore(
        client=client, collection_name="c", embedding=FakeEmbeddings()
    )

    ids = store.add_texts((text for text in TEXTS), batch_size=1)

    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert client.count("c").count == 3


def test_qdrant_vector_store_add_texts_generator_with_explicit_ids() -> None:
    """Explicit ids must be honoured for generator input."""
    client = QdrantClient(":memory:")
    _create_collection(client, "c")
    store = QdrantVectorStore(
        client=client, collection_name="c", embedding=FakeEmbeddings()
    )
    provided_ids = [str(uuid.uuid4()) for _ in TEXTS]

    ids = store.add_texts((text for text in TEXTS), ids=provided_ids)

    assert ids == provided_ids
    assert client.count("c").count == 3


def test_qdrant_vector_store_add_texts_list_still_stores_all_texts() -> None:
    """List input must keep working exactly as before."""
    client = QdrantClient(":memory:")
    _create_collection(client, "c")
    store = QdrantVectorStore(
        client=client, collection_name="c", embedding=FakeEmbeddings()
    )

    ids = store.add_texts(TEXTS)

    assert len(ids) == 3
    assert client.count("c").count == 3


def test_qdrant_add_texts_generator_stores_all_texts() -> None:
    """The legacy ``Qdrant`` sync path must store generator input too."""
    client = QdrantClient(":memory:")
    _create_collection(client, "c")
    store = Qdrant(client, "c", embeddings=FakeEmbeddings())

    ids = store.add_texts(text for text in TEXTS)

    assert len(ids) == 3
    assert client.count("c").count == 3


async def test_qdrant_agenerate_rest_batches_generator_yields_all_texts() -> None:
    """The legacy ``Qdrant`` async batch generator must keep generator input.

    ``aadd_texts`` consumes ``_agenerate_rest_batches``; the generator is
    exercised directly because the async client cannot run against a local
    (``:memory:``) instance.
    """
    client = QdrantClient(":memory:")
    store = Qdrant(client, "c", embeddings=FakeEmbeddings())

    batches = [
        batch
        async for batch in store._agenerate_rest_batches(
            (text for text in TEXTS), batch_size=2
        )
    ]

    ids = [point_id for batch_ids, _ in batches for point_id in batch_ids]
    points = [point for _, batch_points in batches for point in batch_points]
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert len(points) == 3
