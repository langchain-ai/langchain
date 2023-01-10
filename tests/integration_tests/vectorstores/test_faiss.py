"""Test FAISS functionality."""
from typing import List

import pytest

from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.wikipedia import Wikipedia
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.faiss import FAISS


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [[i] * 10 for i in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [0] * 10


def test_faiss() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
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
    expected_document = Document(
        page_content="foo", metadata={"_similarity_to_query": 0.0}
    )
    assert output == [expected_document]


def test_faiss_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    output = docsearch.similarity_search("foo", k=1)
    document = Document(
        page_content="foo", metadata={"page": 0, "_similarity_to_query": 0.0}
    )
    assert output == [document]


def test_faiss_search_not_found() -> None:
    """Test what happens when document is not found."""
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    # Get rid of the docstore to purposefully induce errors.
    docsearch.docstore = InMemoryDocstore({})
    with pytest.raises(ValueError):
        docsearch.similarity_search("foo")


def test_faiss_add_texts() -> None:
    """Test end to end adding of texts."""
    # Create initial doc store.
    texts = ["foo", "bar", "baz"]
    docsearch = FAISS.from_texts(texts, FakeEmbeddings())
    # Test adding a similar document as before.
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    expected_doc = Document(page_content="foo", metadata={"_similarity_to_query": 0.0})
    assert output == [expected_doc, expected_doc]


def test_faiss_add_texts_not_supported() -> None:
    """Test adding of texts to a docstore that doesn't support it."""
    docsearch = FAISS(FakeEmbeddings().embed_query, None, Wikipedia(), {})
    with pytest.raises(ValueError):
        docsearch.add_texts(["foo"])
