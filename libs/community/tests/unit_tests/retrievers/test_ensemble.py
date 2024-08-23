import pytest
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings


@pytest.mark.requires("rank_bm25")
def test_ensemble_retriever_get_relevant_docs() -> None:
    doc_list = [
        "I like apples",
        "I like oranges",
        "Apples and oranges are fruits",
    ]

    from langchain_community.retrievers import BM25Retriever

    dummy_retriever = BM25Retriever.from_texts(doc_list)
    dummy_retriever.k = 1

    ensemble_retriever = EnsembleRetriever(  # type: ignore[call-arg]
        retrievers=[dummy_retriever, dummy_retriever]
    )
    docs = ensemble_retriever.invoke("I like apples")
    assert len(docs) == 1


@pytest.mark.requires("rank_bm25")
def test_weighted_reciprocal_rank() -> None:
    doc1 = Document(page_content="1")
    doc2 = Document(page_content="2")

    from langchain_community.retrievers import BM25Retriever

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


@pytest.mark.requires("rank_bm25", "sklearn")
def test_ensemble_retriever_get_relevant_docs_with_multiple_retrievers() -> None:
    doc_list_a = [
        "I like apples",
        "I like oranges",
        "Apples and oranges are fruits",
    ]
    doc_list_b = [
        "I like melons",
        "I like pineapples",
        "Melons and pineapples are fruits",
    ]
    doc_list_c = [
        "I like avocados",
        "I like strawberries",
        "Avocados and strawberries are fruits",
    ]

    from langchain_community.retrievers import (
        BM25Retriever,
        KNNRetriever,
        TFIDFRetriever,
    )

    dummy_retriever = BM25Retriever.from_texts(doc_list_a)
    dummy_retriever.k = 1
    tfidf_retriever = TFIDFRetriever.from_texts(texts=doc_list_b)
    tfidf_retriever.k = 1
    knn_retriever = KNNRetriever.from_texts(
        texts=doc_list_c, embeddings=FakeEmbeddings(size=100)
    )
    knn_retriever.k = 1

    ensemble_retriever = EnsembleRetriever(
        retrievers=[dummy_retriever, tfidf_retriever, knn_retriever],
        weights=[0.6, 0.3, 0.1],
    )
    docs = ensemble_retriever.invoke("I like apples")
    assert len(docs) == 3
