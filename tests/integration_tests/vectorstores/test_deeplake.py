"""Test Deep Lake functionality."""
import deeplake
import pytest
from pytest import FixtureRequest

from langchain.docstore.document import Document
from langchain.vectorstores import DeepLake
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.fixture
def deeplake_datastore() -> DeepLake:
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = DeepLake.from_texts(
        dataset_path="mem://test_path",
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddings(),
    )
    return docsearch


@pytest.fixture(params=["L1", "L2", "max", "cos"])
def distance_metric(request: FixtureRequest) -> str:
    return request.param


def test_deeplake() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = DeepLake.from_texts(
        dataset_path="mem://test_path", texts=texts, embedding=FakeEmbeddings()
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_deeplake_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = DeepLake.from_texts(
        dataset_path="mem://test_path",
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_deeplakewith_persistence() -> None:
    """Test end to end construction and search, with persistence."""
    dataset_path = "./tests/persist_dir"
    if deeplake.exists(dataset_path):
        deeplake.delete(dataset_path)

    texts = ["foo", "bar", "baz"]
    docsearch = DeepLake.from_texts(
        dataset_path=dataset_path,
        texts=texts,
        embedding=FakeEmbeddings(),
    )

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    # Get a new VectorStore from the persisted directory
    docsearch = DeepLake(
        dataset_path=dataset_path,
        embedding_function=FakeEmbeddings(),
    )
    output = docsearch.similarity_search("foo", k=1)

    # Clean up
    docsearch.delete_dataset()

    # Persist doesn't need to be called again
    # Data will be automatically persisted on object deletion
    # Or on program exit


def test_deeplake_overwrite_flag() -> None:
    """Test overwrite behavior"""
    dataset_path = "./tests/persist_dir"
    if deeplake.exists(dataset_path):
        deeplake.delete(dataset_path)

    texts = ["foo", "bar", "baz"]
    docsearch = DeepLake.from_texts(
        dataset_path=dataset_path,
        texts=texts,
        embedding=FakeEmbeddings(),
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    # Get a new VectorStore from the persisted directory, with no overwrite (implicit)
    docsearch = DeepLake(
        dataset_path=dataset_path,
        embedding_function=FakeEmbeddings(),
    )
    output = docsearch.similarity_search("foo", k=1)
    # assert page still present
    assert output == [Document(page_content="foo")]

    # Get a new VectorStore from the persisted directory, with no overwrite (explicit)
    docsearch = DeepLake(
        dataset_path=dataset_path,
        embedding_function=FakeEmbeddings(),
        overwrite=False,
    )
    output = docsearch.similarity_search("foo", k=1)
    # assert page still present
    assert output == [Document(page_content="foo")]

    # Get a new VectorStore from the persisted directory, with overwrite
    docsearch = DeepLake(
        dataset_path=dataset_path,
        embedding_function=FakeEmbeddings(),
        overwrite=True,
    )
    with pytest.raises(ValueError):
        output = docsearch.similarity_search("foo", k=1)


def test_similarity_search(deeplake_datastore: DeepLake, distance_metric: str) -> None:
    """Test similarity search."""
    output = deeplake_datastore.similarity_search(
        "foo", k=1, distance_metric=distance_metric
    )
    assert output == [Document(page_content="foo", metadata={"page": "0"})]
    deeplake_datastore.delete_dataset()


def test_similarity_search_by_vector(
    deeplake_datastore: DeepLake, distance_metric: str
) -> None:
    """Test similarity search by vector."""
    embeddings = FakeEmbeddings().embed_documents(["foo", "bar", "baz"])
    output = deeplake_datastore.similarity_search_by_vector(
        embeddings[1], k=1, distance_metric=distance_metric
    )
    assert output == [Document(page_content="bar", metadata={"page": "1"})]
    deeplake_datastore.delete_dataset()


def test_similarity_search_with_score(
    deeplake_datastore: DeepLake, distance_metric: str
) -> None:
    """Test similarity search with score."""
    output, score = deeplake_datastore.similarity_search_with_score(
        "foo", k=1, distance_metric=distance_metric
    )[0]
    assert output == Document(page_content="foo", metadata={"page": "0"})
    if distance_metric == "cos":
        assert score == 1.0
    else:
        assert score == 0.0
    deeplake_datastore.delete_dataset()


def test_similarity_search_with_filter(
    deeplake_datastore: DeepLake, distance_metric: str
) -> None:
    """Test similarity search."""

    output = deeplake_datastore.similarity_search(
        "foo",
        k=1,
        distance_metric=distance_metric,
        filter={"metadata": {"page": "1"}},
    )
    assert output == [Document(page_content="bar", metadata={"page": "1"})]
    deeplake_datastore.delete_dataset()


def test_max_marginal_relevance_search(deeplake_datastore: DeepLake) -> None:
    """Test max marginal relevance search by vector."""

    output = deeplake_datastore.max_marginal_relevance_search("foo", k=1, fetch_k=2)

    assert output == [Document(page_content="foo", metadata={"page": "0"})]

    embeddings = FakeEmbeddings().embed_documents(["foo", "bar", "baz"])
    output = deeplake_datastore.max_marginal_relevance_search_by_vector(
        embeddings[0], k=1, fetch_k=2
    )

    assert output == [Document(page_content="foo", metadata={"page": "0"})]
    deeplake_datastore.delete_dataset()


def test_delete_dataset_by_ids(deeplake_datastore: DeepLake) -> None:
    """Test delete dataset."""
    id = deeplake_datastore.vectorstore.dataset.id.data()["value"][0]
    deeplake_datastore.delete(ids=[id])
    assert (
        deeplake_datastore.similarity_search(
            "foo", k=1, filter={"metadata": {"page": "0"}}
        )
        == []
    )
    assert len(deeplake_datastore.vectorstore) == 2

    deeplake_datastore.delete_dataset()


def test_delete_dataset_by_filter(deeplake_datastore: DeepLake) -> None:
    """Test delete dataset."""
    deeplake_datastore.delete(filter={"metadata": {"page": "1"}})
    assert (
        deeplake_datastore.similarity_search(
            "bar", k=1, filter={"metadata": {"page": "1"}}
        )
        == []
    )
    assert len(deeplake_datastore.vectorstore.dataset) == 2

    deeplake_datastore.delete_dataset()


def test_delete_by_path(deeplake_datastore: DeepLake) -> None:
    """Test delete dataset."""
    path = deeplake_datastore.dataset_path
    DeepLake.force_delete_by_path(path)
    assert not deeplake.exists(path)
