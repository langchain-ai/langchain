"""Integration test for PubMed API Wrapper."""

from typing import List

import pytest
from langchain_core.documents import Document

from langchain_community.retrievers import PubMedRetriever


@pytest.fixture
def retriever() -> PubMedRetriever:
    return PubMedRetriever()  # type: ignore[call-arg]


def assert_docs(docs: List[Document]) -> None:
    for doc in docs:
        assert doc.metadata
        assert set(doc.metadata) == {
            "Copyright Information",
            "uid",
            "Title",
            "Published",
        }


def test_load_success(retriever: PubMedRetriever) -> None:
    docs = retriever.invoke("chatgpt")
    assert len(docs) == 3
    assert_docs(docs)


def test_load_success_top_k_results(retriever: PubMedRetriever) -> None:
    retriever.top_k_results = 2
    docs = retriever.invoke("chatgpt")
    assert len(docs) == 2
    assert_docs(docs)


def test_load_no_result(retriever: PubMedRetriever) -> None:
    docs = retriever.invoke("1605.08386WWW")
    assert not docs
