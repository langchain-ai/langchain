from typing import List

import pytest

from langchain.document_loaders.arxiv import ArxivLoader
from langchain.schema import Document


def assert_docs(docs: List[Document]) -> None:
    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        assert set(doc.metadata) == {"Published", "Title", "Authors", "Summary"}


def test_load_success() -> None:
    """Test that returns one document"""
    loader = ArxivLoader(query="1605.08386", load_max_docs=2)

    docs = loader.load()
    assert len(docs) == 1
    print(docs[0].metadata)
    print(docs[0].page_content)
    assert_docs(docs)


def test_load_returns_no_result() -> None:
    """Test that returns no docs"""
    loader = ArxivLoader(query="1605.08386WWW", load_max_docs=2)
    docs = loader.load()

    assert len(docs) == 0


def test_load_returns_limited_docs() -> None:
    """Test that returns several docs"""
    expected_docs = 2
    loader = ArxivLoader(query="ChatGPT", load_max_docs=expected_docs)
    docs = loader.load()

    assert len(docs) == expected_docs
    assert_docs(docs)


def test_load_returns_full_set_of_metadata() -> None:
    """Test that returns several docs"""
    loader = ArxivLoader(query="ChatGPT", load_max_docs=1, load_all_available_meta=True)
    docs = loader.load()
    assert len(docs) == 1
    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        assert set(doc.metadata).issuperset(
            {"Published", "Title", "Authors", "Summary"}
        )
        print(doc.metadata)
        assert len(set(doc.metadata)) > 4


@pytest.mark.skip(reason="test could be flaky")
def test_load_issue_9046() -> None:
    """Test for the fixed issue 9046"""
    expected_docs = 3

    # ":" character could not be an issue
    loader = ArxivLoader(
        query="MetaGPT: Meta Programming for Multi-Agent Collaborative Framework",
        load_max_docs=expected_docs,
    )
    docs = loader.load()

    assert_docs(docs)
    assert "MetaGPT" in docs[0].metadata["Title"]

    # "-" character could not be an issue
    loader = ArxivLoader(
        query="MetaGPT - Meta Programming for Multi-Agent Collaborative Framework",
        load_max_docs=expected_docs,
    )
    docs = loader.load()

    assert_docs(docs)
    assert "MetaGPT" in docs[0].metadata["Title"]
