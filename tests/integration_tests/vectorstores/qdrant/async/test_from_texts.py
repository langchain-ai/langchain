import uuid
from typing import Optional

import pytest

from langchain.schema import Document
from langchain.vectorstores import Qdrant
from langchain.vectorstores.qdrant import QdrantException
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)

from .common import qdrant_is_not_running

# Skipping all the tests in the module if Qdrant is not running on localhost.
pytestmark = pytest.mark.skipif(
    qdrant_is_not_running(), reason="Qdrant server is not running"
)


@pytest.mark.asyncio
async def test_qdrant_from_texts_stores_duplicated_texts() -> None:
    """Test end to end Qdrant.afrom_texts stores duplicated texts separately."""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex

    await Qdrant.afrom_texts(
        ["abc", "abc"],
        ConsistentFakeEmbeddings(),
        collection_name=collection_name,
    )

    client = QdrantClient()
    assert 2 == client.count(collection_name).count


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
async def test_qdrant_from_texts_stores_ids(
    batch_size: int, vector_name: Optional[str]
) -> None:
    """Test end to end Qdrant.afrom_texts stores provided ids."""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex
    ids = [
        "fa38d572-4c31-4579-aedc-1960d79df6df",
        "cdc1aa36-d6ab-4fb2-8a94-56674fd27484",
    ]
    await Qdrant.afrom_texts(
        ["abc", "def"],
        ConsistentFakeEmbeddings(),
        ids=ids,
        collection_name=collection_name,
        batch_size=batch_size,
        vector_name=vector_name,
    )

    client = QdrantClient()
    assert 2 == client.count(collection_name).count
    stored_ids = [point.id for point in client.scroll(collection_name)[0]]
    assert set(ids) == set(stored_ids)


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_name", ["custom-vector"])
async def test_qdrant_from_texts_stores_embeddings_as_named_vectors(
    vector_name: str,
) -> None:
    """Test end to end Qdrant.afrom_texts stores named vectors if name is provided."""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex

    await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        vector_name=vector_name,
    )

    client = QdrantClient()
    assert 5 == client.count(collection_name).count
    assert all(
        vector_name in point.vector  # type: ignore[operator]
        for point in client.scroll(collection_name, with_vectors=True)[0]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_name", [None, "custom-vector"])
async def test_qdrant_from_texts_reuses_same_collection(
    vector_name: Optional[str],
) -> None:
    """Test if Qdrant.afrom_texts reuses the same collection"""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex
    embeddings = ConsistentFakeEmbeddings()

    await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        embeddings,
        collection_name=collection_name,
        vector_name=vector_name,
    )

    await Qdrant.afrom_texts(
        ["foo", "bar"],
        embeddings,
        collection_name=collection_name,
        vector_name=vector_name,
    )

    client = QdrantClient()
    assert 7 == client.count(collection_name).count


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_name", [None, "custom-vector"])
async def test_qdrant_from_texts_raises_error_on_different_dimensionality(
    vector_name: Optional[str],
) -> None:
    """Test if Qdrant.afrom_texts raises an exception if dimensionality does not
    match"""
    collection_name = uuid.uuid4().hex

    await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(dimensionality=10),
        collection_name=collection_name,
        vector_name=vector_name,
    )

    with pytest.raises(QdrantException):
        await Qdrant.afrom_texts(
            ["foo", "bar"],
            ConsistentFakeEmbeddings(dimensionality=5),
            collection_name=collection_name,
            vector_name=vector_name,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ["first_vector_name", "second_vector_name"],
    [
        (None, "custom-vector"),
        ("custom-vector", None),
        ("my-first-vector", "my-second_vector"),
    ],
)
async def test_qdrant_from_texts_raises_error_on_different_vector_name(
    first_vector_name: Optional[str],
    second_vector_name: Optional[str],
) -> None:
    """Test if Qdrant.afrom_texts raises an exception if vector name does not match"""
    collection_name = uuid.uuid4().hex

    await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(dimensionality=10),
        collection_name=collection_name,
        vector_name=first_vector_name,
    )

    with pytest.raises(QdrantException):
        await Qdrant.afrom_texts(
            ["foo", "bar"],
            ConsistentFakeEmbeddings(dimensionality=5),
            collection_name=collection_name,
            vector_name=second_vector_name,
        )


@pytest.mark.asyncio
async def test_qdrant_from_texts_raises_error_on_different_distance() -> None:
    """Test if Qdrant.afrom_texts raises an exception if distance does not match"""
    collection_name = uuid.uuid4().hex

    await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(dimensionality=10),
        collection_name=collection_name,
        distance_func="Cosine",
    )

    with pytest.raises(QdrantException):
        await Qdrant.afrom_texts(
            ["foo", "bar"],
            ConsistentFakeEmbeddings(dimensionality=5),
            collection_name=collection_name,
            distance_func="Euclid",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_name", [None, "custom-vector"])
async def test_qdrant_from_texts_recreates_collection_on_force_recreate(
    vector_name: Optional[str],
) -> None:
    """Test if Qdrant.afrom_texts recreates the collection even if config mismatches"""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex

    await Qdrant.afrom_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(dimensionality=10),
        collection_name=collection_name,
        vector_name=vector_name,
    )

    await Qdrant.afrom_texts(
        ["foo", "bar"],
        ConsistentFakeEmbeddings(dimensionality=5),
        collection_name=collection_name,
        vector_name=vector_name,
        force_recreate=True,
    )

    client = QdrantClient()
    assert 2 == client.count(collection_name).count


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
async def test_qdrant_from_texts_stores_metadatas(
    batch_size: int, content_payload_key: str, metadata_payload_key: str
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
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]
