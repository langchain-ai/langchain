"""Integration test for PubMed API Wrapper."""
from typing import List

import pytest

from langchain.document_loaders import PubMedLoader
from langchain.schema import Document

xmltodict = pytest.importorskip("xmltodict")


def test_load_success() -> None:
    """Test that returns the correct answer"""
    api_client = PubMedLoader(query="chatgpt")
    docs = api_client.load()
    print(docs)
    assert len(docs) == api_client.load_max_docs == 3
    assert_docs(docs)


def test_load_success_load_max_docs() -> None:
    """Test that returns the correct answer"""
    api_client = PubMedLoader(query="chatgpt", load_max_docs=2)
    docs = api_client.load()
    print(docs)
    assert len(docs) == api_client.load_max_docs == 2
    assert_docs(docs)


def test_load_returns_no_result() -> None:
    """Test that gives no result."""
    api_client = PubMedLoader(query="1605.08386WWW")
    docs = api_client.load()
    assert len(docs) == 0


def test_load_no_content() -> None:
    """Returns a Document without content."""
    api_client = PubMedLoader(query="37548971")
    docs = api_client.load()
    print(docs)
    assert len(docs) > 0
    assert docs[0].page_content == ""


def assert_docs(docs: List[Document]) -> None:
    for doc in docs:
        assert doc.metadata
        assert set(doc.metadata) == {
            "Copyright Information",
            "uid",
            "Title",
            "Published",
        }
