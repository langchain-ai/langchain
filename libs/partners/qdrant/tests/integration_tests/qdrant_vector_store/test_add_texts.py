import uuid
from typing import Union

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    ConsistentFakeSparseEmbeddings,
    assert_documents_equals,
)
from tests.integration_tests.fixtures import qdrant_locations, retrieval_modes


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
def test_qdrant_add_documents_extends_existing_collection(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    new_texts = ["foobar", "foobaz"]
    docsearch.add_documents([Document(page_content=content) for content in new_texts])
    output = docsearch.similarity_search("foobar", k=1)
    assert_documents_equals(output, [Document(page_content="foobar")])


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
@pytest.mark.parametrize("batch_size", [1, 64])
def test_qdrant_add_texts_returns_all_ids(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
    batch_size: int,
) -> None:
    """Test end to end Qdrant.add_texts returns unique ids."""
    docsearch = QdrantVectorStore.from_texts(
        ["foobar"],
        ConsistentFakeEmbeddings(),
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        batch_size=batch_size,
    )

    ids = docsearch.add_texts(["foo", "bar", "baz"])
    assert 3 == len(ids)
    assert 3 == len(set(ids))
    assert 3 == len(docsearch.get_by_ids(ids))


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
def test_qdrant_add_texts_stores_duplicated_texts(
    location: str,
    vector_name: str,
) -> None:
    """Test end to end Qdrant.add_texts stores duplicated texts separately."""

    client = QdrantClient(location)
    collection_name = uuid.uuid4().hex
    vectors_config = {
        vector_name: models.VectorParams(size=10, distance=models.Distance.COSINE)
    }
    client.recreate_collection(collection_name, vectors_config=vectors_config)

    vec_store = QdrantVectorStore(
        client,
        collection_name,
        embedding=ConsistentFakeEmbeddings(),
        vector_name=vector_name,
    )
    ids = vec_store.add_texts(["abc", "abc"], [{"a": 1}, {"a": 2}])

    assert 2 == len(set(ids))
    assert 2 == client.count(collection_name).count


@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
@pytest.mark.parametrize("retrieval_mode", retrieval_modes())
@pytest.mark.parametrize(
    "sparse_vector_name", ["my-sparse-vector", "another-sparse-vector"]
)
@pytest.mark.parametrize("batch_size", [1, 64])
def test_qdrant_add_texts_stores_ids(
    location: str,
    vector_name: str,
    retrieval_mode: RetrievalMode,
    sparse_vector_name: str,
    batch_size: int,
) -> None:
    """Test end to end Qdrant.add_texts stores provided ids."""
    ids: list[Union[str, int]] = [
        "fa38d572-4c31-4579-aedc-1960d79df6df",
        432,
        432145435,
    ]
    collection_name = uuid.uuid4().hex
    vec_store = QdrantVectorStore.from_texts(
        ["abc", "def", "ghi"],
        ConsistentFakeEmbeddings(),
        ids=ids,
        collection_name=collection_name,
        location=location,
        vector_name=vector_name,
        retrieval_mode=retrieval_mode,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        batch_size=batch_size,
    )

    assert 3 == vec_store.client.count(collection_name).count
    stored_ids = [point.id for point in vec_store.client.scroll(collection_name)[0]]
    assert set(ids) == set(stored_ids)
    assert 3 == len(vec_store.get_by_ids(ids))
