import pytest

from langchain.retrievers.bm25 import BM25Retriever
from langchain.schema import Document


@pytest.mark.requires("gensim")
def test_from_texts() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    bm25_retriever = BM25Retriever.from_texts(texts=input_texts)
    assert len(bm25_retriever.docs) == 3
    assert len(bm25_retriever.dictionary) == 7


@pytest.mark.requires("gensim")
def test_from_texts_with_proprecess_filters() -> None:
    from gensim.parsing.preprocessing import strip_punctuation, remove_stopwords
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    bm25_retriever = BM25Retriever.from_texts(
        texts=input_texts, 
        preprocess_filters=[lambda x: x.lower(), strip_punctuation, remove_stopwords])
    assert len(bm25_retriever.docs) == 3
    # After removing stopwords only (pen, bag) left
    assert len(bm25_retriever.dictionary) == 2


@pytest.mark.requires("gensim")
def test_from_documents() -> None:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
    ]
    bm25_retriever = BM25Retriever.from_documents(documents=input_docs)
    assert len(bm25_retriever.docs) == 3
    assert len(bm25_retriever.dictionary) == 7

