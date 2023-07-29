import pytest

from langchain.retrievers.bm25 import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.schema import Document


@pytest.mark.requires("rank_bm25")
def test_ensemble_retriever_get_relevant_docs() -> None:
    doc_list = [
        "I like apples",
        "I like oranges",
        "Apples and oranges are fruits",
    ]

    dummy_retriever = BM25Retriever.from_texts(doc_list)
    dummy_retriever.k = 1

    ensemble_retriever = EnsembleRetriever(
        retrievers=[dummy_retriever, dummy_retriever]
    )
    docs = ensemble_retriever.get_relevant_documents("I like apples")
    assert len(docs) == 1


@pytest.mark.requires("rank_bm25")
def test_weighted_reciprocal_rank() -> None:
    doc1 = Document(page_content="1")
    doc2 = Document(page_content="2")

    dummy_retriever = BM25Retriever.from_texts(["1", "2"])
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dummy_retriever, dummy_retriever], weights=[0.4, 0.5], c=0
    )
    result = ensemble_retriever.weighted_reciprocal_rank([[doc1, doc2], [doc2, doc1]])
    assert result[0].page_content == "2"
    assert result[1].page_content == "1"

    ensemble_retriever.weights = [0.5, 0.4]
    result = ensemble_retriever.weighted_reciprocal_rank([[doc1, doc2], [doc2, doc1]])
    assert result[0].page_content == "1"
    assert result[1].page_content == "2"
