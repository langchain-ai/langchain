from __future__ import annotations

import os
import uuid
from typing import Optional

import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document

from langchain_qdrant import Qdrant
from langchain_qdrant.vectorstores import QdrantException
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    assert_documents_equals,
)
from tests.integration_tests.fixtures import (
    qdrant_locations,
)


@pytest.mark.parametrize("qdrant_location", qdrant_locations())
async def test_qdrant_from_texts_stores_duplicated_texts(qdrant_location: str) -> None:
    """Test end to end Qdrant.afrom_texts stores duplicated texts separately."""
    collection_name = uuid.uuid4().hex

    vec_store = await Qdrant.afrom_texts(
        ["abc", "abc"],
        ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        location=qdrant_location,
    )

    client = vec_store.client
    assert client.count(collection_name).count == 2


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
@pytest.mark.parametrize("qdrant_location", qdrant_locations())
async def test_qdrant_from_texts_stores_ids(
    batch_size: int, vector_name: Optional[str], qdrant_location: str
) -> None:
    """Test end to end Qdrant.afrom_texts stores provided ids."""
    collection_name = uuid.uuid4().hex
    ids = [
        "fa38d572-4c31-4579-aedc-1960d79df6df",
        "cdc1aa36-d6ab-4fb2-8a94-56674fd27484",
    ]
    vec_store = await Qdrant.afrom_texts(
        ["abc", "def"],
        ConsistentFakeEmbeddings(),
        ids=ids,
        collection_name=collection_name,
        batch_size=batch_size,
        vector_name=vector_name,
        location=qdrant_location,
    )

    client = vec_store.client
    assert client.count(collection_name).count == 2
    stored_ids = [point.id for point in client.scroll(collection_name)[0]]
    assert set(ids) == set(stored_ids)


@pytest.mark.parametrize("vector_name", ["custom-vector"])
@pytest.mark.parametrize("qdrant_location", qdrant_locations())
async def test_qdrant_from_texts_stores_embeddings_as_named_vectors(
    vector_name: str,
    qdrant_location: str,
) -> None:
    """Test end to end Qdrant.afrom_texts stores named vectors if name is provided."""
    collection_name = uuid.uuid4().hex

    vec_store = await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        vector_name=vector_name,
        location=qdrant_location,
    )

    client = vec_store.client
    assert client.count(collection_name).count == 5
    assert all(
        vector_name in point.vector  # type: ignore[operator]
        for point in client.scroll(collection_name, with_vectors=True)[0]
    )


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
@pytest.mark.parametrize("vector_name", [None, "custom-vector"])
async def test_qdrant_from_texts_reuses_same_collection(
    location: str, vector_name: Optional[str]
) -> None:
    """Test if Qdrant.afrom_texts reuses the same collection."""
    collection_name = uuid.uuid4().hex
    embeddings = ConsistentFakeEmbeddings()

    await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        embeddings,
        collection_name=collection_name,
        vector_name=vector_name,
        location=location,
    )

    vec_store = await Qdrant.afrom_texts(
        ["foo", "bar"],
        embeddings,
        collection_name=collection_name,
        vector_name=vector_name,
        location=location,
    )

    client = vec_store.client
    assert client.count(collection_name).count == 7


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
@pytest.mark.parametrize("vector_name", [None, "custom-vector"])
async def test_qdrant_from_texts_raises_error_on_different_dimensionality(
    location: str,
    vector_name: Optional[str],
) -> None:
    """Test if Qdrant.afrom_texts raises an exception if dimensionality does not
    match.
    """
    collection_name = uuid.uuid4().hex

    await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(dimensionality=10),
        collection_name=collection_name,
        vector_name=vector_name,
        location=location,
    )

    with pytest.raises(QdrantException):
        await Qdrant.afrom_texts(
            ["foo", "bar"],
            ConsistentFakeEmbeddings(dimensionality=5),
            collection_name=collection_name,
            vector_name=vector_name,
            location=location,
        )


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
@pytest.mark.parametrize(
    ("first_vector_name", "second_vector_name"),
    [
        (None, "custom-vector"),
        ("custom-vector", None),
        ("my-first-vector", "my-second_vector"),
    ],
)
async def test_qdrant_from_texts_raises_error_on_different_vector_name(
    location: str,
    first_vector_name: Optional[str],
    second_vector_name: Optional[str],
) -> None:
    """Test if Qdrant.afrom_texts raises an exception if vector name does not match."""
    collection_name = uuid.uuid4().hex

    await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(dimensionality=10),
        collection_name=collection_name,
        vector_name=first_vector_name,
        location=location,
    )

    with pytest.raises(QdrantException):
        await Qdrant.afrom_texts(
            ["foo", "bar"],
            ConsistentFakeEmbeddings(dimensionality=5),
            collection_name=collection_name,
            vector_name=second_vector_name,
            location=location,
        )


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
async def test_qdrant_from_texts_raises_error_on_different_distance(
    location: str,
) -> None:
    """Test if Qdrant.afrom_texts raises an exception if distance does not match."""
    collection_name = uuid.uuid4().hex

    await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(dimensionality=10),
        collection_name=collection_name,
        distance_func="Cosine",
        location=location,
    )

    with pytest.raises(QdrantException):
        await Qdrant.afrom_texts(
            ["foo", "bar"],
            ConsistentFakeEmbeddings(dimensionality=5),
            collection_name=collection_name,
            distance_func="Euclid",
            location=location,
        )


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
@pytest.mark.parametrize("vector_name", [None, "custom-vector"])
async def test_qdrant_from_texts_recreates_collection_on_force_recreate(
    location: str,
    vector_name: Optional[str],
) -> None:
    """Test if Qdrant.afrom_texts recreates the collection even if config mismatches."""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex

    await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(dimensionality=10),
        collection_name=collection_name,
        vector_name=vector_name,
        location=location,
    )

    await Qdrant.afrom_texts(
        ["foo", "bar"],
        ConsistentFakeEmbeddings(dimensionality=5),
        collection_name=collection_name,
        vector_name=vector_name,
        force_recreate=True,
        location=location,
    )

    client = QdrantClient(location=location, api_key=os.getenv("QDRANT_API_KEY"))
    assert client.count(collection_name).count == 2
    vector_params = client.get_collection(collection_name).config.params.vectors
    if vector_name is not None:
        vector_params = vector_params[vector_name]  # type: ignore[index]
    assert vector_params.size == 5  # type: ignore[union-attr]


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
@pytest.mark.parametrize("qdrant_location", qdrant_locations())
async def test_qdrant_from_texts_stores_metadatas(
    batch_size: int,
    content_payload_key: str,
    metadata_payload_key: str,
    qdrant_location: str,
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = await Qdrant.afrom_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
        location=qdrant_location,
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert_documents_equals(
        output, [Document(page_content="foo", metadata={"page": 0})]
    )
