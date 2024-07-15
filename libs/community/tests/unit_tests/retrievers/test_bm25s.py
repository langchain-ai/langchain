import pytest
from langchain_community.retrievers.bm25s import BM25SRetriever
from langchain_core.documents import Document


@pytest.mark.requires("bm25s")
def test_from_texts() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    bm25_retriever = BM25SRetriever.from_texts(texts=input_texts)
    assert len(bm25_retriever.docs) == 3
    assert set(bm25_retriever.retriever.vocab_dict.keys()) == {
        "do",
        "you",
        "have",
        "bag",
        "pen",
    }


@pytest.mark.requires("bm25s")
def test_from_texts_with_bm25_params() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    bm25_retriever = BM25SRetriever.from_texts(
        texts=input_texts, bm25s_params={"method": "robertson"}
    )
    # should count only multiple words (have, pan)
    assert bm25_retriever.retriever.method == "robertson"


@pytest.mark.requires("bm25s")
def test_from_documents() -> None:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
    ]
    bm25_retriever = BM25SRetriever.from_documents(documents=input_docs)
    assert len(bm25_retriever.docs) == 3
    assert set(bm25_retriever.retriever.vocab_dict.keys()) == {
        "do",
        "you",
        "have",
        "bag",
        "pen",
    }


@pytest.mark.requires("bm25s")
def test_repr() -> None:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
    ]
    bm25_retriever = BM25SRetriever.from_documents(documents=input_docs)
    assert "I have a pen" not in repr(bm25_retriever)
