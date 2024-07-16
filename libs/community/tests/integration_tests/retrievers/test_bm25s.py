"""Test Azure AI Search wrapper."""

import pytest
from langchain_community.retrievers.bm25s import BM25SRetriever
from langchain_core.documents import Document


@pytest.mark.requires("bm25s")
@pytest.fixture
def retriever() -> BM25SRetriever:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
    ]

    bm25_retriever = BM25SRetriever.from_documents(documents=input_docs)
    return bm25_retriever


def test_invoke(retriever: BM25SRetriever) -> None:
    expected = [Document(page_content="I have a pen.")]

    retriever.k = 1
    results = retriever.invoke("I have a pen!")

    assert len(results) == retriever.k
    assert results == expected
