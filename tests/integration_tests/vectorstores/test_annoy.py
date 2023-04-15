"""Test Annoy functionality."""
import tempfile

import pytest

from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores.annoy import Annoy
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_annoy() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Annoy.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(page_content="foo"),
            index_to_id[1]: Document(page_content="bar"),
            index_to_id[2]: Document(page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_annoy_vector_sim() -> None:
    """Test vector similarity."""
    texts = ["foo", "bar", "baz"]
    docsearch = Annoy.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(page_content="foo"),
            index_to_id[1]: Document(page_content="bar"),
            index_to_id[2]: Document(page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    query_vec = FakeEmbeddings().embed_query(text="foo")
    output = docsearch.similarity_search_by_vector(query_vec, k=1)
    assert output == [Document(page_content="foo")]

    # make sure we can have k > docstore size
    output = docsearch.max_marginal_relevance_search_by_vector(query_vec, k=10)
    assert len(output) == len(texts)


def test_annoy_vector_sim_by_index() -> None:
    """Test vector similarity."""
    texts = ["foo", "bar", "baz"]
    docsearch = Annoy.from_texts(texts, FakeEmbeddings())
    index_to_id = docsearch.index_to_docstore_id
    expected_docstore = InMemoryDocstore(
        {
            index_to_id[0]: Document(page_content="foo"),
            index_to_id[1]: Document(page_content="bar"),
            index_to_id[2]: Document(page_content="baz"),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search_by_index(2, k=1)
    assert output == [Document(page_content="baz")]


def test_annoy_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Annoy.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    expected_docstore = InMemoryDocstore(
        {
            docsearch.index_to_docstore_id[0]: Document(
                page_content="foo", metadata={"page": 0}
            ),
            docsearch.index_to_docstore_id[1]: Document(
                page_content="bar", metadata={"page": 1}
            ),
            docsearch.index_to_docstore_id[2]: Document(
                page_content="baz", metadata={"page": 2}
            ),
        }
    )
    assert docsearch.docstore.__dict__ == expected_docstore.__dict__
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_annoy_search_not_found() -> None:
    """Test what happens when document is not found."""
    texts = ["foo", "bar", "baz"]
    docsearch = Annoy.from_texts(texts, FakeEmbeddings())
    # Get rid of the docstore to purposefully induce errors.
    docsearch.docstore = InMemoryDocstore({})

    with pytest.raises(ValueError):
        docsearch.similarity_search("foo")


def test_annoy_add_texts() -> None:
    """Test end to end adding of texts."""
    # Create initial doc store.
    texts = ["foo", "bar", "baz"]
    docsearch = Annoy.from_texts(texts, FakeEmbeddings())
    # Test adding a similar document as before.
    with pytest.raises(NotImplementedError):
        docsearch.add_texts(["foo"])


def test_annoy_local_save_load() -> None:
    """Test end to end serialization."""
    texts = ["foo", "bar", "baz"]
    docsearch = Annoy.from_texts(texts, FakeEmbeddings())

    temp_dir = tempfile.TemporaryDirectory()
    docsearch.save_local(temp_dir.name)
    loaded_docsearch = Annoy.load_local(temp_dir.name, FakeEmbeddings())

    assert docsearch.index_to_docstore_id == loaded_docsearch.index_to_docstore_id
    assert docsearch.docstore.__dict__ == loaded_docsearch.docstore.__dict__
    assert loaded_docsearch.index is not None
