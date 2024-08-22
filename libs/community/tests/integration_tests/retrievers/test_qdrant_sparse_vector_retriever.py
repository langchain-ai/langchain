import random
import uuid
from typing import List, Tuple

import pytest
from langchain_core.documents import Document

from langchain_community.retrievers import QdrantSparseVectorRetriever
from langchain_community.vectorstores.qdrant import QdrantException


def consistent_fake_sparse_encoder(
    query: str, size: int = 100, density: float = 0.7
) -> Tuple[List[int], List[float]]:
    """
    Generates a consistent fake sparse vector.

    Parameters:
    - query (str): The query string to make the function deterministic.
    - size (int): The size of the vector to generate.
    - density (float): The density of the vector to generate.

    Returns:
    - indices (list): List of indices where the non-zero elements are located.
    - values (list): List of corresponding float values at the non-zero indices.
    """
    # Ensure density is within the valid range [0, 1]
    density = max(0.0, min(1.0, density))

    # Use a deterministic seed based on the query
    seed = hash(query)
    random.seed(seed)

    # Calculate the number of non-zero elements based on density
    num_non_zero_elements = int(size * density)

    # Generate random indices without replacement
    indices = sorted(random.sample(range(size), num_non_zero_elements))

    # Generate random float values for the non-zero elements
    values = [random.uniform(0.0, 1.0) for _ in range(num_non_zero_elements)]

    return indices, values


@pytest.fixture
def retriever() -> QdrantSparseVectorRetriever:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(location=":memory:")

    collection_name = uuid.uuid4().hex
    vector_name = uuid.uuid4().hex

    client.recreate_collection(
        collection_name,
        vectors_config={},
        sparse_vectors_config={
            vector_name: models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        },
    )

    return QdrantSparseVectorRetriever(
        client=client,
        collection_name=collection_name,
        sparse_vector_name=vector_name,
        sparse_encoder=consistent_fake_sparse_encoder,
    )


def test_invalid_collection_name(retriever: QdrantSparseVectorRetriever) -> None:
    with pytest.raises(QdrantException) as e:
        QdrantSparseVectorRetriever(
            client=retriever.client,
            collection_name="invalid collection",
            sparse_vector_name=retriever.sparse_vector_name,
            sparse_encoder=consistent_fake_sparse_encoder,
        )
    assert "does not exist" in str(e.value)


def test_invalid_sparse_vector_name(retriever: QdrantSparseVectorRetriever) -> None:
    with pytest.raises(QdrantException) as e:
        QdrantSparseVectorRetriever(
            client=retriever.client,
            collection_name=retriever.collection_name,
            sparse_vector_name="invalid sparse vector",
            sparse_encoder=consistent_fake_sparse_encoder,
        )

    assert "does not contain sparse vector" in str(e.value)


def test_add_documents(retriever: QdrantSparseVectorRetriever) -> None:
    documents = [
        Document(page_content="hello world", metadata={"a": 1}),
        Document(page_content="foo bar", metadata={"b": 2}),
        Document(page_content="baz qux", metadata={"c": 3}),
    ]

    ids = retriever.add_documents(documents)

    assert retriever.client.count(retriever.collection_name, exact=True).count == 3

    documents = [
        Document(page_content="hello world"),
        Document(page_content="foo bar"),
        Document(page_content="baz qux"),
    ]

    ids = retriever.add_documents(documents)

    assert len(ids) == 3

    assert retriever.client.count(retriever.collection_name, exact=True).count == 6


def test_add_texts(retriever: QdrantSparseVectorRetriever) -> None:
    retriever.add_texts(
        ["hello world", "foo bar", "baz qux"], [{"a": 1}, {"b": 2}, {"c": 3}]
    )

    assert retriever.client.count(retriever.collection_name, exact=True).count == 3

    retriever.add_texts(["hello world", "foo bar", "baz qux"])

    assert retriever.client.count(retriever.collection_name, exact=True).count == 6


def test_invoke(retriever: QdrantSparseVectorRetriever) -> None:
    retriever.add_texts(["Hai there!", "Hello world!", "Foo bar baz!"])

    expected = [Document(page_content="Hai there!")]

    retriever.k = 1
    results = retriever.invoke("Hai there!")

    assert len(results) == retriever.k
    assert results == expected
    assert retriever.invoke("Hai there!") == expected


def test_invoke_with_filter(
    retriever: QdrantSparseVectorRetriever,
) -> None:
    from qdrant_client import models

    retriever.add_texts(
        ["Hai there!", "Hello world!", "Foo bar baz!"],
        [
            {"value": 1},
            {"value": 2},
            {"value": 3},
        ],
    )

    retriever.filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.value", match=models.MatchValue(value=2)
            )
        ]
    )
    results = retriever.invoke("Some query")

    assert results[0] == Document(page_content="Hello world!", metadata={"value": 2})
