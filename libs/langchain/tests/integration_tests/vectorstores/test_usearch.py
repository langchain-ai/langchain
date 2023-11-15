"""Test USearch functionality."""
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.usearch import USearch
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_usearch_from_texts() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = USearch.from_texts(texts, FakeEmbeddings())
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_usearch_from_documents() -> None:
    """Test from_documents constructor."""
    texts = ["foo", "bar", "baz"]
    docs = [Document(page_content=t, metadata={"a": "b"}) for t in texts]
    docsearch = USearch.from_documents(docs, FakeEmbeddings())
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"a": "b"})]


def test_usearch_add_texts() -> None:
    """Test adding a new document"""
    texts = ["foo", "bar", "baz"]
    docsearch = USearch.from_texts(texts, FakeEmbeddings())
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2)
    assert output == [Document(page_content="foo"), Document(page_content="foo")]


def test_ip() -> None:
    """Test inner product distance."""
    texts = ["foo", "bar", "baz"]
    docsearch = USearch.from_texts(texts, FakeEmbeddings(), metric="ip")
    output = docsearch.similarity_search_with_score("far", k=2)
    _, score = output[1]
    assert score == -8.0


def test_l2() -> None:
    """Test Flat L2 distance."""
    texts = ["foo", "bar", "baz"]
    docsearch = USearch.from_texts(texts, FakeEmbeddings(), metric="l2_sq")
    output = docsearch.similarity_search_with_score("far", k=2)
    _, score = output[1]
    assert score == 1.0


def test_cos() -> None:
    """Test cosine distance."""
    texts = ["foo", "bar", "baz"]
    docsearch = USearch.from_texts(texts, FakeEmbeddings(), metric="cos")
    output = docsearch.similarity_search_with_score("far", k=2)
    _, score = output[1]
    assert score == pytest.approx(0.05, abs=0.002)
