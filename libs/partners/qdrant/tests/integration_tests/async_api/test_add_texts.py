from __future__ import annotations

import os
import uuid
from typing import Optional

import pytest  # type: ignore[import-not-found]

from langchain_qdrant import Qdrant
from tests.integration_tests.common import ConsistentFakeEmbeddings
from tests.integration_tests.fixtures import qdrant_locations

API_KEY = os.getenv("QDRANT_API_KEY")


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("qdrant_location", qdrant_locations())
async def test_qdrant_aadd_texts_returns_all_ids(
    batch_size: int, qdrant_location: str
) -> None:
    """Test end to end Qdrant.aadd_texts returns unique ids."""
    docsearch: Qdrant = Qdrant.from_texts(
        ["foobar"],
        ConsistentFakeEmbeddings(),
        batch_size=batch_size,
        location=qdrant_location,
    )

    ids = await docsearch.aadd_texts(["foo", "bar", "baz"])
    assert len(ids) == 3
    assert len(set(ids)) == 3


@pytest.mark.parametrize("vector_name", [None, "my-vector"])
@pytest.mark.parametrize("qdrant_location", qdrant_locations())
async def test_qdrant_aadd_texts_stores_duplicated_texts(
    vector_name: Optional[str], qdrant_location: str
) -> None:
    """Test end to end Qdrant.aadd_texts stores duplicated texts separately."""
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest

    client = QdrantClient(location=qdrant_location, api_key=API_KEY)
    collection_name = uuid.uuid4().hex
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

    assert len(set(ids)) == 2
    assert client.count(collection_name).count == 2


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("qdrant_location", qdrant_locations())
async def test_qdrant_aadd_texts_stores_ids(
    batch_size: int, qdrant_location: str
) -> None:
    """Test end to end Qdrant.aadd_texts stores provided ids."""
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest

    ids = [
        "fa38d572-4c31-4579-aedc-1960d79df6df",
        "cdc1aa36-d6ab-4fb2-8a94-56674fd27484",
    ]

    client = QdrantClient(location=qdrant_location, api_key=API_KEY)
    collection_name = uuid.uuid4().hex
    client.recreate_collection(
        collection_name,
        vectors_config=rest.VectorParams(size=10, distance=rest.Distance.COSINE),
    )

    vec_store = Qdrant(client, collection_name, ConsistentFakeEmbeddings())
    returned_ids = await vec_store.aadd_texts(
        ["abc", "def"], ids=ids, batch_size=batch_size
    )

    assert all(
        first == second for first, second in zip(ids, returned_ids, strict=False)
    )
    assert client.count(collection_name).count == 2
    stored_ids = [point.id for point in client.scroll(collection_name)[0]]
    assert set(ids) == set(stored_ids)


@pytest.mark.parametrize("vector_name", ["custom-vector"])
@pytest.mark.parametrize("qdrant_location", qdrant_locations())
async def test_qdrant_aadd_texts_stores_embeddings_as_named_vectors(
    vector_name: str, qdrant_location: str
) -> None:
    """Test end to end Qdrant.aadd_texts stores named vectors if name is provided."""
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest

    collection_name = uuid.uuid4().hex

    client = QdrantClient(location=qdrant_location, api_key=API_KEY)
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

    assert client.count(collection_name).count == 5
    assert all(
        vector_name in point.vector  # type: ignore[operator]
        for point in client.scroll(collection_name, with_vectors=True)[0]
    )
