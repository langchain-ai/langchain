"""Test Qdrant functionality."""
import tempfile
from typing import Callable, Optional

import pytest
from qdrant_client.http import models as rest

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Qdrant
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
def test_qdrant_similarity_search(
    batch_size: int, content_payload_key: str, metadata_payload_key: str
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=":memory:",
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
def test_qdrant_similarity_search_by_vector(
    batch_size: int, content_payload_key: str, metadata_payload_key: str
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=":memory:",
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
    )
    embeddings = ConsistentFakeEmbeddings().embed_query("foo")
    output = docsearch.similarity_search_by_vector(embeddings, k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
def test_qdrant_similarity_search_with_score_by_vector(
    batch_size: int, content_payload_key: str, metadata_payload_key: str
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=":memory:",
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        batch_size=batch_size,
    )
    embeddings = ConsistentFakeEmbeddings().embed_query("foo")
    output = docsearch.similarity_search_with_score_by_vector(embeddings, k=1)
    assert len(output) == 1
    document, score = output[0]
    assert document == Document(page_content="foo")
    assert score >= 0


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
def test_qdrant_add_documents(batch_size: int, vector_name: Optional[str]) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch: Qdrant = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        location=":memory:",
        batch_size=batch_size,
        vector_name=vector_name,
    )

    new_texts = ["foobar", "foobaz"]
    docsearch.add_documents(
        [Document(page_content=content) for content in new_texts], batch_size=batch_size
    )
    output = docsearch.similarity_search("foobar", k=1)
    # StatefulFakeEmbeddings return the same query embedding as the first document
    # embedding computed in `embedding.embed_documents`. Thus, "foo" embedding is the
    # same as "foobar" embedding
    assert output == [Document(page_content="foobar")] or output == [
        Document(page_content="foo")
    ]


@pytest.mark.parametrize("batch_size", [1, 64])
def test_qdrant_add_texts_returns_all_ids(batch_size: int) -> None:
    """Test end to end Qdrant.add_texts returns unique ids."""
    docsearch: Qdrant = Qdrant.from_texts(
        ["foobar"],
        ConsistentFakeEmbeddings(),
        location=":memory:",
        batch_size=batch_size,
    )

    ids = docsearch.add_texts(["foo", "bar", "baz"])
    assert 3 == len(ids)
    assert 3 == len(set(ids))


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "foo"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "bar"])
def test_qdrant_with_metadatas(
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
    assert output == [Document(page_content="foo", metadata={"page": 0})]


@pytest.mark.parametrize("batch_size", [1, 64])
def test_qdrant_similarity_search_filters(batch_size: int) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "metadata": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        location=":memory:",
        batch_size=batch_size,
    )

    output = docsearch.similarity_search(
        "foo", k=1, filter={"page": 1, "metadata": {"page": 2, "pages": [3]}}
    )
    assert output == [
        Document(
            page_content="bar",
            metadata={"page": 1, "metadata": {"page": 2, "pages": [3, -1]}},
        )
    ]


@pytest.mark.parametrize("vector_name", [None, "my-vector"])
def test_qdrant_similarity_search_with_relevance_score_no_threshold(
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "metadata": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        location=":memory:",
        vector_name=vector_name,
    )
    output = docsearch.similarity_search_with_relevance_scores(
        "foo", k=3, score_threshold=None
    )
    assert len(output) == 3
    for i in range(len(output)):
        assert round(output[i][1], 2) >= 0
        assert round(output[i][1], 2) <= 1


def test_qdrant_similarity_search_with_relevance_score_with_threshold() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "metadata": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        location=":memory:",
    )

    score_threshold = 0.98
    kwargs = {"score_threshold": score_threshold}
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3, **kwargs)
    assert len(output) == 1
    assert all([score >= score_threshold for _, score in output])


def test_qdrant_similarity_search_with_relevance_score_with_threshold_and_filter() -> (
    None
):
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "metadata": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        location=":memory:",
    )
    score_threshold = 0.99  # for almost exact match
    # test negative filter condition
    negative_filter = {"page": 1, "metadata": {"page": 2, "pages": [3]}}
    kwargs = {"filter": negative_filter, "score_threshold": score_threshold}
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3, **kwargs)
    assert len(output) == 0
    # test positive filter condition
    positive_filter = {"page": 0, "metadata": {"page": 1, "pages": [2]}}
    kwargs = {"filter": positive_filter, "score_threshold": score_threshold}
    output = docsearch.similarity_search_with_relevance_scores("foo", k=3, **kwargs)
    assert len(output) == 1
    assert all([score >= score_threshold for _, score in output])


def test_qdrant_similarity_search_filters_with_qdrant_filters() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [
        {"page": i, "details": {"page": i + 1, "pages": [i + 2, -1]}}
        for i in range(len(texts))
    ]
    docsearch = Qdrant.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        location=":memory:",
    )

    qdrant_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="metadata.page",
                match=rest.MatchValue(value=1),
            ),
            rest.FieldCondition(
                key="metadata.details.page",
                match=rest.MatchValue(value=2),
            ),
            rest.FieldCondition(
                key="metadata.details.pages",
                match=rest.MatchAny(any=[3]),
            ),
        ]
    )
    output = docsearch.similarity_search("foo", k=1, filter=qdrant_filter)
    assert output == [
        Document(
            page_content="bar",
            metadata={"page": 1, "details": {"page": 2, "pages": [3, -1]}},
        )
    ]


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("content_payload_key", [Qdrant.CONTENT_KEY, "test_content"])
@pytest.mark.parametrize("metadata_payload_key", [Qdrant.METADATA_KEY, "test_metadata"])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
def test_qdrant_max_marginal_relevance_search(
    batch_size: int,
    content_payload_key: str,
    metadata_payload_key: str,
    vector_name: Optional[str],
) -> None:
    """Test end to end construction and MRR search."""
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
        vector_name=vector_name,
    )
    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)
    assert output == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
    ]


@pytest.mark.parametrize(
    ["embeddings", "embedding_function"],
    [
        (ConsistentFakeEmbeddings(), None),
        (ConsistentFakeEmbeddings().embed_query, None),
        (None, ConsistentFakeEmbeddings().embed_query),
    ],
)
def test_qdrant_embedding_interface(
    embeddings: Optional[Embeddings], embedding_function: Optional[Callable]
) -> None:
    """Test Qdrant may accept different types for embeddings."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    collection_name = "test"

    Qdrant(
        client,
        collection_name,
        embeddings=embeddings,
        embedding_function=embedding_function,
    )


@pytest.mark.parametrize(
    ["embeddings", "embedding_function"],
    [
        (ConsistentFakeEmbeddings(), ConsistentFakeEmbeddings().embed_query),
        (None, None),
    ],
)
def test_qdrant_embedding_interface_raises_value_error(
    embeddings: Optional[Embeddings], embedding_function: Optional[Callable]
) -> None:
    """Test Qdrant requires only one method for embeddings."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    collection_name = "test"

    with pytest.raises(ValueError):
        Qdrant(
            client,
            collection_name,
            embeddings=embeddings,
            embedding_function=embedding_function,
        )


@pytest.mark.parametrize("vector_name", [None, "my-vector"])
def test_qdrant_add_texts_stores_duplicated_texts(vector_name: Optional[str]) -> None:
    """Test end to end Qdrant.add_texts stores duplicated texts separately."""
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest

    client = QdrantClient(":memory:")
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
    ids = vec_store.add_texts(["abc", "abc"], [{"a": 1}, {"a": 2}])

    assert 2 == len(set(ids))
    assert 2 == client.count(collection_name).count


def test_qdrant_from_texts_stores_duplicated_texts() -> None:
    """Test end to end Qdrant.from_texts stores duplicated texts separately."""
    from qdrant_client import QdrantClient

    with tempfile.TemporaryDirectory() as tmpdir:
        vec_store = Qdrant.from_texts(
            ["abc", "abc"],
            ConsistentFakeEmbeddings(),
            collection_name="test",
            path=str(tmpdir),
        )
        del vec_store

        client = QdrantClient(path=str(tmpdir))
        assert 2 == client.count("test").count


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vector_name", [None, "my-vector"])
def test_qdrant_from_texts_stores_ids(
    batch_size: int, vector_name: Optional[str]
) -> None:
    """Test end to end Qdrant.from_texts stores provided ids."""
    from qdrant_client import QdrantClient

    with tempfile.TemporaryDirectory() as tmpdir:
        ids = [
            "fa38d572-4c31-4579-aedc-1960d79df6df",
            "cdc1aa36-d6ab-4fb2-8a94-56674fd27484",
        ]
        vec_store = Qdrant.from_texts(
            ["abc", "def"],
            ConsistentFakeEmbeddings(),
            ids=ids,
            collection_name="test",
            path=str(tmpdir),
            batch_size=batch_size,
            vector_name=vector_name,
        )
        del vec_store

        client = QdrantClient(path=str(tmpdir))
        assert 2 == client.count("test").count
        stored_ids = [point.id for point in client.scroll("test")[0]]
        assert set(ids) == set(stored_ids)


@pytest.mark.parametrize("batch_size", [1, 64])
def test_qdrant_add_texts_stores_ids(batch_size: int) -> None:
    """Test end to end Qdrant.add_texts stores provided ids."""
    from qdrant_client import QdrantClient

    ids = [
        "fa38d572-4c31-4579-aedc-1960d79df6df",
        "cdc1aa36-d6ab-4fb2-8a94-56674fd27484",
    ]

    client = QdrantClient(":memory:")
    collection_name = "test"
    client.recreate_collection(
        collection_name,
        vectors_config=rest.VectorParams(size=10, distance=rest.Distance.COSINE),
    )

    vec_store = Qdrant(client, "test", ConsistentFakeEmbeddings())
    returned_ids = vec_store.add_texts(["abc", "def"], ids=ids, batch_size=batch_size)

    assert all(first == second for first, second in zip(ids, returned_ids))
    assert 2 == client.count("test").count
    stored_ids = [point.id for point in client.scroll("test")[0]]
    assert set(ids) == set(stored_ids)


@pytest.mark.parametrize("vector_name", ["custom-vector"])
def test_qdrant_from_texts_stores_embeddings_as_named_vectors(vector_name: str) -> None:
    """Test end to end Qdrant.from_texts stores named vectors if name is provided."""
    from qdrant_client import QdrantClient

    collection_name = "test"
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
        assert 5 == client.count("test").count
        assert all(
            vector_name in point.vector  # type: ignore[operator]
            for point in client.scroll(collection_name, with_vectors=True)[0]
        )


@pytest.mark.parametrize("vector_name", ["custom-vector"])
def test_qdrant_add_texts_stores_embeddings_as_named_vectors(vector_name: str) -> None:
    """Test end to end Qdrant.add_texts stores named vectors if name is provided."""
    from qdrant_client import QdrantClient

    collection_name = "test"

    client = QdrantClient(":memory:")
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
    vec_store.add_texts(["lorem", "ipsum", "dolor", "sit", "amet"])

    assert 5 == client.count("test").count
    assert all(
        vector_name in point.vector  # type: ignore[operator]
        for point in client.scroll(collection_name, with_vectors=True)[0]
    )
