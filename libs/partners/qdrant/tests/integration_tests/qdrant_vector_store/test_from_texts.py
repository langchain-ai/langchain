import uuid
from typing import List, Union

import pytest
from langchain_core.documents import Document
from qdrant_client import models

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.qdrant import QdrantVectorStoreError
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    ConsistentFakeSparseEmbeddings,
    assert_documents_equals,
)
from tests.integration_tests.fixtures import qdrant_locations, retrieval_modes


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
def test_vectorstore_from_texts(location: str, retrieval_mode: RetrievalMode) -> None:
    """Test end to end Qdrant.from_texts stores texts."""
    collection_name = uuid.uuid4().hex

    vec_store = QdrantVectorStore.from_texts(
        ["Lorem ipsum dolor sit amet", "Ipsum dolor sit amet"],
        ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        location=location,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    assert 2 == vec_store.client.count(collection_name).count


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
def test_qdrant_from_texts_stores_ids(
    batch_size: int,
    vector_name: str,
    sparse_vector_name: str,
    location: str,
    retrieval_mode: RetrievalMode,
) -> None:
    """Test end to end Qdrant.from_texts stores provided ids."""
    collection_name = uuid.uuid4().hex
    ids: List[Union[str, int]] = [
        "fa38d572-4c31-4579-aedc-1960d79df6df",
        786,
    ]
    vec_store = QdrantVectorStore.from_texts(
        ["abc", "def"],
        ConsistentFakeEmbeddings(),
        ids=ids,
        collection_name=collection_name,
        location=location,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        batch_size=batch_size,
        vector_name=vector_name,
        sparse_vector_name=sparse_vector_name,
    )

    assert 2 == vec_store.client.count(collection_name).count
    stored_ids = [point.id for point in vec_store.client.retrieve(collection_name, ids)]
    assert set(ids) == set(stored_ids)


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
def test_qdrant_from_texts_stores_embeddings_as_named_vectors(
    location: str,
    retrieval_mode: RetrievalMode,
    vector_name: str,
    sparse_vector_name: str,
) -> None:
    """Test end to end Qdrant.from_texts stores named vectors if name is provided."""

    collection_name = uuid.uuid4().hex
    vec_store = QdrantVectorStore.from_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    assert 5 == vec_store.client.count(collection_name).count
    if retrieval_mode in retrieval_modes(sparse=False):
        assert all(
            (vector_name in point.vector or isinstance(point.vector, list))  # type: ignore
            for point in vec_store.client.scroll(collection_name, with_vectors=True)[0]
        )
    if retrieval_mode in retrieval_modes(dense=False):
        assert all(
            sparse_vector_name in point.vector  # type: ignore
            for point in vec_store.client.scroll(collection_name, with_vectors=True)[0]
        )


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
def test_qdrant_from_texts_reuses_same_collection(
    location: str,
    retrieval_mode: RetrievalMode,
    vector_name: str,
    sparse_vector_name: str,
) -> None:
    """Test if Qdrant.from_texts reuses the same collection"""
    collection_name = uuid.uuid4().hex
    embeddings = ConsistentFakeEmbeddings()
    sparse_embeddings = ConsistentFakeSparseEmbeddings()
    vec_store = QdrantVectorStore.from_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        embeddings,
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=sparse_embeddings,
    )
    del vec_store

    vec_store = QdrantVectorStore.from_texts(
        ["foo", "bar"],
        embeddings,
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=sparse_embeddings,
    )

    assert 7 == vec_store.client.count(collection_name).count


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes(sparse=False))
def test_qdrant_from_texts_raises_error_on_different_dimensionality(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
) -> None:
    """Test if Qdrant.from_texts raises an exception if dimensionality does not match"""
    collection_name = uuid.uuid4().hex
    QdrantVectorStore.from_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(dimensionality=10),
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    with pytest.raises(QdrantVectorStoreError) as excinfo:
        QdrantVectorStore.from_texts(
            ["foo", "bar"],
            ConsistentFakeEmbeddings(dimensionality=5),
            collection_name=collection_name,
            location=location,
            vector_name=vector_name,
            retrieval_mode=retrieval_mode,
            sparse_embedding=ConsistentFakeSparseEmbeddings(),
        )

        expected_message = "collection is configured for dense vectors "
        "with 10 dimensions. Selected embeddings are 5-dimensional"
        assert expected_message in str(excinfo.value)


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
@pytest.mark.parametrize(
    ["first_vector_name", "second_vector_name"],
    [
        ("", "custom-vector"),
        ("custom-vector", ""),
        ("my-first-vector", "my-second_vector"),
    ],
)
@pytest.mark.parametrize("retrieval_mode", retrieval_modes(sparse=False))
def test_qdrant_from_texts_raises_error_on_different_vector_name(
    location: str,
    first_vector_name: str,
    second_vector_name: str,
    retrieval_mode: RetrievalMode,
) -> None:
    """Test if Qdrant.from_texts raises an exception if vector name does not match"""
    collection_name = uuid.uuid4().hex
    QdrantVectorStore.from_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(dimensionality=10),
        collection_name=collection_name,
        location=location,
        vector_name=first_vector_name,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    with pytest.raises(QdrantVectorStoreError) as excinfo:
        QdrantVectorStore.from_texts(
            ["foo", "bar"],
            ConsistentFakeEmbeddings(dimensionality=10),
            collection_name=collection_name,
            location=location,
            vector_name=second_vector_name,
            retrieval_mode=retrieval_mode,
            sparse_embedding=ConsistentFakeSparseEmbeddings(),
        )

        expected_message = "does not contain dense vector named"
        assert expected_message in str(excinfo.value)


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes(sparse=False))
def test_qdrant_from_texts_raises_error_on_different_distance(
    location: str, vector_name: str, retrieval_mode: RetrievalMode
) -> None:
    """Test if Qdrant.from_texts raises an exception if distance does not match"""
    collection_name = uuid.uuid4().hex
    QdrantVectorStore.from_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(),
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        distance=models.Distance.COSINE,
        retrieval_mode=retrieval_mode,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    with pytest.raises(QdrantVectorStoreError) as excinfo:
        QdrantVectorStore.from_texts(
            ["foo", "bar"],
            ConsistentFakeEmbeddings(),
            collection_name=collection_name,
            location=location,
            vector_name=vector_name,
            distance=models.Distance.EUCLID,
            retrieval_mode=retrieval_mode,
            sparse_embedding=ConsistentFakeSparseEmbeddings(),
        )

        expected_message = "configured for COSINE similarity, but requested EUCLID"
        assert expected_message in str(excinfo.value)


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
def test_qdrant_from_texts_recreates_collection_on_force_recreate(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
) -> None:
    collection_name = uuid.uuid4().hex
    vec_store = QdrantVectorStore.from_texts(
        ["lorem", "ipsum", "dolor", "sit", "amet"],
        ConsistentFakeEmbeddings(dimensionality=10),
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    vec_store = QdrantVectorStore.from_texts(
        ["foo", "bar"],
        ConsistentFakeEmbeddings(dimensionality=5),
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        force_recreate=True,
    )

    assert 2 == vec_store.client.count(collection_name).count


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("content_payload_key", [QdrantVectorStore.CONTENT_KEY, "foo"])
@pytest.mark.parametrize(
    "metadata_payload_key", [QdrantVectorStore.METADATA_KEY, "bar"]
)
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
def test_qdrant_from_texts_stores_metadatas(
    location: str,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
) -> None:
    """Test end to end construction and search."""
    texts = ["fabrin", "barizda"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        location=location,
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )
    output = docsearch.similarity_search("fabrin", k=1)
    assert_documents_equals(
        output, [Document(page_content="fabrin", metadata={"page": 0})]
    )


@pytest.mark.parametrize("location", qdrant_locations(use_in_memory=False))
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes(sparse=False))
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
def test_from_texts_passed_optimizers_config_and_on_disk_payload(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
) -> None:
    collection_name = uuid.uuid4().hex
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    optimizers_config = models.OptimizersConfigDiff(memmap_threshold=1000)
    vec_store = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        collection_create_options={
            "on_disk_payload": True,
            "optimizers_config": optimizers_config,
        },
        vector_params={
            "on_disk": True,
        },
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    collection_info = vec_store.client.get_collection(collection_name)
    assert collection_info.config.params.vectors[vector_name].on_disk is True  # type: ignore
    assert collection_info.config.optimizer_config.memmap_threshold == 1000
    assert collection_info.config.params.on_disk_payload is True
