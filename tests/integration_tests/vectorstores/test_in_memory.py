import pytest

from langchain.schema import Document
from langchain.vectorstores.in_memory import InMemory
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_docarray_vec_store_init() -> None:
    """Test end to end construction and simple similarity search."""
    texts = ["foo", "bar", "baz"]
    docsearch = InMemory.from_texts(
        texts,
        FakeEmbeddings(),
    )
    assert isinstance(docsearch, InMemory)


@pytest.fixture
def docarray_vec_store():
    texts = ["foo", "bar", "baz"]
    docsearch = InMemory.from_texts(
        texts,
        FakeEmbeddings(),
    )
    return docsearch


def test_sim_search(docarray_vec_store) -> None:
    """Test end to end construction and simple similarity search."""

    output = docarray_vec_store.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_sim_search_with_score(docarray_vec_store) -> None:
    """Test end to end construction and similarity search with score."""

    output = docarray_vec_store.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo"), 1.0)]


def test_sim_search_by_vector(docarray_vec_store):
    """Test end to end construction and similarity search by vector."""
    embedding = [1.0] * 10
    output = docarray_vec_store.similarity_search_by_vector(embedding, k=1)

    assert output == [Document(page_content="bar")]

