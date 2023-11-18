"""Integration test for PubMed API Wrapper."""
from typing import List

import pytest

from langchain.retrievers import PubMedRetriever
from langchain.schema import Document


@pytest.fixture
def retriever() -> PubMedRetriever:
    return PubMedRetriever()


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
    docs = retriever.get_relevant_documents(query="chatgpt")
    assert len(docs) == 3
    assert_docs(docs)


def test_load_success_top_k_results(retriever: PubMedRetriever) -> None:
    retriever.top_k_results = 2
    docs = retriever.get_relevant_documents(query="chatgpt")
    assert len(docs) == 2
    assert_docs(docs)


def test_load_no_result(retriever: PubMedRetriever) -> None:
    docs = retriever.get_relevant_documents("1605.08386WWW")
    assert not docs
