import tempfile
import uuid
from typing import Optional

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores.qdrant import QdrantException
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)
from tests.integration_tests.vectorstores.qdrant.common import (
    assert_documents_equals,
    qdrant_is_not_running,
)


def test_qdrant_from_texts_stores_duplicated_texts() -> None:
    """Test end to end Qdrant.from_texts stores duplicated texts separately."""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex

    with tempfile.TemporaryDirectory() as tmpdir:
        vec_store = Qdrant.from_texts(
            ["abc", "abc"],
            ConsistentFakeEmbeddings(),
            collection_name=collection_name,
            path=str(tmpdir),
        )
        del vec_store

        client = QdrantClient(path=str(tmpdir))
        assert 2 == client.count(collection_name).count


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
def test_qdrant_from_texts_stores_ids(
    batch_size: int, vector_name: Optional[str]
) -> None:
    """Test end to end Qdrant.from_texts stores provided ids."""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmpdir:
        ids = [
            "fa38d572-4c31-4579-aedc-1960d79df6df",
            "cdc1aa36-d6ab-4fb2-8a94-56674fd27484",
        ]
        vec_store = Qdrant.from_texts(
            ["abc", "def"],
            ConsistentFakeEmbeddings(),
            ids=ids,
            collection_name=collection_name,
            path=str(tmpdir),
            batch_size=batch_size,
            vector_name=vector_name,
        )
        del vec_store

        client = QdrantClient(path=str(tmpdir))
        assert 2 == client.count(collection_name).count
        stored_ids = [point.id for point in client.scroll(collection_name)[0]]
        assert set(ids) == set(stored_ids)


@pytest.mark.parametrize("vector_name", ["custom-vector"])
def test_qdrant_from_texts_stores_embeddings_as_named_vectors(vector_name: str) -> None:
    """Test end to end Qdrant.from_texts stores named vectors if name is provided."""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmpdir:
        vec_store = Qdrant.from_texts(
            ["lorem", "ipsum", "dolor", "sit", "amet"],
            ConsistentFakeEmbeddings(),
            collection_name=collection_name,
            path=str(tmpdir),
            vector_name=vector_name,
        )
        del vec_store

        client = QdrantClient(path=str(tmpdir))
        assert 5 == client.count(collection_name).count
        assert all(
            vector_name in point.vector  # type: ignore[operator]
            for point in client.scroll(collection_name, with_vectors=True)[0]
        )


@pytest.mark.parametrize("vector_name", [None, "custom-vector"])
def test_qdrant_from_texts_reuses_same_collection(vector_name: Optional[str]) -> None:
    """Test if Qdrant.from_texts reuses the same collection"""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex
    embeddings = ConsistentFakeEmbeddings()
    with tempfile.TemporaryDirectory() as tmpdir:
        vec_store = Qdrant.from_texts(
            ["lorem", "ipsum", "dolor", "sit", "amet"],
            embeddings,
            collection_name=collection_name,
            path=str(tmpdir),
            vector_name=vector_name,
        )
        del vec_store

        vec_store = Qdrant.from_texts(
            ["foo", "bar"],
            embeddings,
            collection_name=collection_name,
            path=str(tmpdir),
            vector_name=vector_name,
        )
        del vec_store

        client = QdrantClient(path=str(tmpdir))
        assert 7 == client.count(collection_name).count


@pytest.mark.parametrize("vector_name", [None, "custom-vector"])
def test_qdrant_from_texts_raises_error_on_different_dimensionality(
    vector_name: Optional[str],
) -> None:
    """Test if Qdrant.from_texts raises an exception if dimensionality does not match"""
    collection_name = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmpdir:
        vec_store = Qdrant.from_texts(
            ["lorem", "ipsum", "dolor", "sit", "amet"],
            ConsistentFakeEmbeddings(dimensionality=10),
            collection_name=collection_name,
            path=str(tmpdir),
            vector_name=vector_name,
        )
        del vec_store

        with pytest.raises(QdrantException):
            Qdrant.from_texts(
                ["foo", "bar"],
                ConsistentFakeEmbeddings(dimensionality=5),
                collection_name=collection_name,
                path=str(tmpdir),
                vector_name=vector_name,
            )


@pytest.mark.parametrize(
    ["first_vector_name", "second_vector_name"],
    [
        (None, "custom-vector"),
        ("custom-vector", None),
        ("my-first-vector", "my-second_vector"),
    ],
)
def test_qdrant_from_texts_raises_error_on_different_vector_name(
    first_vector_name: Optional[str],
    second_vector_name: Optional[str],
) -> None:
    """Test if Qdrant.from_texts raises an exception if vector name does not match"""
    collection_name = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmpdir:
        vec_store = Qdrant.from_texts(
            ["lorem", "ipsum", "dolor", "sit", "amet"],
            ConsistentFakeEmbeddings(dimensionality=10),
            collection_name=collection_name,
            path=str(tmpdir),
            vector_name=first_vector_name,
        )
        del vec_store

        with pytest.raises(QdrantException):
            Qdrant.from_texts(
                ["foo", "bar"],
                ConsistentFakeEmbeddings(dimensionality=5),
                collection_name=collection_name,
                path=str(tmpdir),
                vector_name=second_vector_name,
            )


def test_qdrant_from_texts_raises_error_on_different_distance() -> None:
    """Test if Qdrant.from_texts raises an exception if distance does not match"""
    collection_name = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmpdir:
        vec_store = Qdrant.from_texts(
            ["lorem", "ipsum", "dolor", "sit", "amet"],
            ConsistentFakeEmbeddings(),
            collection_name=collection_name,
            path=str(tmpdir),
            distance_func="Cosine",
        )
        del vec_store

        with pytest.raises(QdrantException) as excinfo:
            Qdrant.from_texts(
                ["foo", "bar"],
                ConsistentFakeEmbeddings(),
                collection_name=collection_name,
                path=str(tmpdir),
                distance_func="Euclid",
            )

        expected_message = (
            "configured for COSINE similarity, but requested EUCLID. Please set "
            "`distance_func` parameter to `COSINE`"
        )
        assert expected_message in str(excinfo.value)


@pytest.mark.parametrize("vector_name", [None, "custom-vector"])
def test_qdrant_from_texts_recreates_collection_on_force_recreate(
    vector_name: Optional[str],
) -> None:
    """Test if Qdrant.from_texts recreates the collection even if config mismatches"""
    from qdrant_client import QdrantClient

    collection_name = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmpdir:
        vec_store = Qdrant.from_texts(
            ["lorem", "ipsum", "dolor", "sit", "amet"],
            ConsistentFakeEmbeddings(dimensionality=10),
            collection_name=collection_name,
            path=str(tmpdir),
            vector_name=vector_name,
        )
        del vec_store

        vec_store = Qdrant.from_texts(
            ["foo", "bar"],
            ConsistentFakeEmbeddings(dimensionality=5),
            collection_name=collection_name,
            path=str(tmpdir),
            vector_name=vector_name,
            force_recreate=True,
        )
        del vec_store

        client = QdrantClient(path=str(tmpdir))
        assert 2 == client.count(collection_name).count


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
def test_qdrant_from_texts_stores_metadatas(
    batch_size: int, content_payload_key: str, metadata_payload_key: str
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        location=":memory:",
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert_documents_equals(
        output, [Document(page_content="foo", metadata={"page": 0})]
    )


@pytest.mark.skipif(qdrant_is_not_running(), reason="Qdrant is not running")
def test_from_texts_passed_optimizers_config_and_on_disk_payload() -> None:
    from qdrant_client import models

    collection_name = uuid.uuid4().hex
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    optimizers_config = models.OptimizersConfigDiff(memmap_threshold=1000)
    vec_store = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        optimizers_config=optimizers_config,
        on_disk_payload=True,
        on_disk=True,
        collection_name=collection_name,
    )

    collection_info = vec_store.client.get_collection(collection_name)
    assert collection_info.config.params.vectors.on_disk is True
    assert collection_info.config.optimizer_config.memmap_threshold == 1000
    assert collection_info.config.params.on_disk_payload is True
