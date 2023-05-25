import pytest

from langchain.retrievers.tfidf import TFIDFRetriever
from langchain.schema import Document


@pytest.mark.requires("sklearn")
def test_from_texts() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    tfidf_retriever = TFIDFRetriever.from_texts(texts=input_texts)
    assert len(tfidf_retriever.docs) == 3
    assert tfidf_retriever.tfidf_array.toarray().shape == (3, 5)


@pytest.mark.requires("sklearn")
def test_from_texts_with_tfidf_params() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    tfidf_retriever = TFIDFRetriever.from_texts(
        texts=input_texts, tfidf_params={"min_df": 2}
    )
    # should count only multiple words (have, pan)
    assert tfidf_retriever.tfidf_array.toarray().shape == (3, 2)


@pytest.mark.requires("sklearn")
def test_from_documents() -> None:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
    ]
    tfidf_retriever = TFIDFRetriever.from_documents(documents=input_docs)
    assert len(tfidf_retriever.docs) == 3
    assert tfidf_retriever.tfidf_array.toarray().shape == (3, 5)
