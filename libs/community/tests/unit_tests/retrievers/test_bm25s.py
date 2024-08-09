import tempfile
from datetime import datetime

import pytest
from langchain_community.retrievers.bm25s import BM25SRetriever
from langchain_core.documents import Document


@pytest.mark.requires("bm25s")
def test_from_texts() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    bm25_retriever = BM25SRetriever.from_texts(texts=input_texts)
    assert len(bm25_retriever.retriever.corpus) == 3
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
    assert len(bm25_retriever.retriever.corpus) == 3
    assert set(bm25_retriever.retriever.vocab_dict.keys()) == {
        "do",
        "you",
        "have",
        "bag",
        "pen",
    }


@pytest.mark.requires("bm25s")
def test_bm25s_save_load() -> None:
    """Test end to end serialization."""
    texts = ["foo", "bar", "baz"]
    temp_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with tempfile.TemporaryDirectory(suffix="_" + temp_timestamp + "/") as temp_folder:
        _ = BM25SRetriever.from_texts(texts, persist_directory=temp_folder)
        new_docsearch = BM25SRetriever.load(temp_folder)
    assert new_docsearch.retriever.vocab_dict is not None


@pytest.mark.requires("bm25s")
def test_repr() -> None:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
    ]
    bm25_retriever = BM25SRetriever.from_documents(documents=input_docs)
    assert "I have a pen" not in repr(bm25_retriever)
