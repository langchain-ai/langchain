import pytest

from langchain.retrievers.bm25 import BM25Retriever
from langchain.schema import Document


@pytest.mark.requires("rank_bm25")
def test_from_texts() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    bm25_retriever = BM25Retriever.from_texts(texts=input_texts)
    assert len(bm25_retriever.docs) == 3
    assert bm25_retriever.vectorizer.doc_len == [4, 5, 4]


@pytest.mark.requires("rank_bm25")
def test_from_texts_with_bm25_params() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    bm25_retriever = BM25Retriever.from_texts(
        texts=input_texts, bm25_params={"epsilon": 10}
    )
    # should count only multiple words (have, pan)
    assert bm25_retriever.vectorizer.epsilon == 10


@pytest.mark.requires("rank_bm25")
def test_from_documents() -> None:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
    ]
    bm25_retriever = BM25Retriever.from_documents(documents=input_docs)
    assert len(bm25_retriever.docs) == 3
    assert bm25_retriever.vectorizer.doc_len == [4, 5, 4]
