import pytest
from langchain_core.documents import Document

from langchain_community.retrievers.bm25 import BM25Retriever


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


@pytest.mark.requires("rank_bm25")
def test_repr() -> None:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
    ]
    bm25_retriever = BM25Retriever.from_documents(documents=input_docs)
    assert "I have a pen" not in repr(bm25_retriever)


@pytest.mark.requires("rank_bm25")
def test_doc_id() -> None:
    docs_with_ids = [
        Document(page_content="I have a pen.", id="1"),
        Document(page_content="Do you have a pen?", id="2"),
        Document(page_content="I have a bag.", id="3"),
    ]
    docs_without_ids = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
    ]
    docs_with_some_ids = [
        Document(page_content="I have a pen.", id="1"),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag.", id="3"),
    ]
    bm25_retriever_with_ids = BM25Retriever.from_documents(documents=docs_with_ids)
    bm25_retriever_without_ids = BM25Retriever.from_documents(
        documents=docs_without_ids
    )
    bm25_retriever_with_some_ids = BM25Retriever.from_documents(
        documents=docs_with_some_ids
    )
    for doc in bm25_retriever_with_ids.docs:
        assert doc.id is not None
    for doc in bm25_retriever_without_ids.docs:
        assert doc.id is None
    for doc in bm25_retriever_with_some_ids.docs:
        if doc.page_content == "I have a pen.":
            assert doc.id == "1"
        elif doc.page_content == "Do you have a pen?":
            assert doc.id is None
        elif doc.page_content == "I have a bag.":
            assert doc.id == "3"
        else:
            raise ValueError("Unexpected document")
