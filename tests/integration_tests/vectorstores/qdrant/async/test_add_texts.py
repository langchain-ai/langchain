from typing import Optional

import pytest
from qdrant_client.http import models as rest

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
async def test_qdrant_aadd_texts_returns_all_ids(batch_size: int) -> None:
    """Test end to end Qdrant.aadd_texts returns unique ids."""
    docsearch: Qdrant = Qdrant.from_texts(
        ["foobar"],
        ConsistentFakeEmbeddings(),
        batch_size=batch_size,
    )

    ids = await docsearch.aadd_texts(["foo", "bar", "baz"])
    assert 3 == len(ids)
    assert 3 == len(set(ids))


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
async def test_qdrant_aadd_texts_stores_duplicated_texts(
    vector_name: Optional[str],
) -> None:
    """Test end to end Qdrant.aadd_texts stores duplicated texts separately."""
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest

    client = QdrantClient()
    collection_name = "test"
    vectors_config = rest.VectorParams(size=10, distance=rest.Distance.COSINE)
    if vector_name is not None:
        vectors_config = {vector_name: vectors_config}  # type: ignore[assignment]
    client.recreate_collection(collection_name, vectors_config=vectors_config)

    vec_store = Qdrant(
        client,
        collection_name,
        embeddings=ConsistentFakeEmbeddings(),
        vector_name=vector_name,
    )
    ids = await vec_store.aadd_texts(["abc", "abc"], [{"a": 1}, {"a": 2}])

    assert 2 == len(set(ids))
    assert 2 == client.count(collection_name).count


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 64])
async def test_qdrant_aadd_texts_stores_ids(batch_size: int) -> None:
    """Test end to end Qdrant.aadd_texts stores provided ids."""
    from qdrant_client import QdrantClient

    ids = [
        "fa38d572-4c31-4579-aedc-1960d79df6df",
        "cdc1aa36-d6ab-4fb2-8a94-56674fd27484",
    ]

    client = QdrantClient()
    collection_name = "test"
    client.recreate_collection(
        collection_name,
        vectors_config=rest.VectorParams(size=10, distance=rest.Distance.COSINE),
    )

    vec_store = Qdrant(client, collection_name, ConsistentFakeEmbeddings())
    returned_ids = await vec_store.aadd_texts(
        ["abc", "def"], ids=ids, batch_size=batch_size
    )

    assert all(first == second for first, second in zip(ids, returned_ids))
    assert 2 == client.count(collection_name).count
    stored_ids = [point.id for point in client.scroll(collection_name)[0]]
    assert set(ids) == set(stored_ids)


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_name", ["custom-vector"])
async def test_qdrant_aadd_texts_stores_embeddings_as_named_vectors(
    vector_name: str,
) -> None:
    """Test end to end Qdrant.aadd_texts stores named vectors if name is provided."""
    from qdrant_client import QdrantClient

    collection_name = "test"

    client = QdrantClient()
    client.recreate_collection(
        collection_name,
        vectors_config={
            vector_name: rest.VectorParams(size=10, distance=rest.Distance.COSINE)
        },
    )

    vec_store = Qdrant(
        client,
        collection_name,
        ConsistentFakeEmbeddings(),
        vector_name=vector_name,
    )
    await vec_store.aadd_texts(["lorem", "ipsum", "dolor", "sit", "amet"])

    assert 5 == client.count(collection_name).count
    assert all(
        vector_name in point.vector  # type: ignore[operator]
        for point in client.scroll(collection_name, with_vectors=True)[0]
    )
