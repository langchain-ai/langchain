"""Test Clarifai vectore store functionality."""
import time

from langchain.docstore.document import Document
from langchain.vectorstores import Clarifai


def test_clarifai_with_from_texts() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    USER_ID = "minhajul"
    APP_ID = "test-lang-2"
    NUMBER_OF_DOCS = 1
    docsearch = Clarifai.from_texts(
        user_id=USER_ID,
        app_id=APP_ID,
        texts=texts,
        pat=None,
        number_of_docs=NUMBER_OF_DOCS,
    )
    time.sleep(2.5)
    output = docsearch.similarity_search("foo")
    assert output == [Document(page_content="foo")]


def test_clarifai_with_from_documents() -> None:
    """Test end to end construction and search."""
    # Initial document content and id
    initial_content = "foo"

    # Create an instance of Document with initial content and metadata
    original_doc = Document(page_content=initial_content, metadata={"page": "0"})
    USER_ID = "minhajul"
    APP_ID = "test-lang-2"
    NUMBER_OF_DOCS = 1
    docsearch = Clarifai.from_documents(
        user_id=USER_ID,
        app_id=APP_ID,
        documents=[original_doc],
        pat=None,
        number_of_docs=NUMBER_OF_DOCS,
    )
    time.sleep(2.5)
    output = docsearch.similarity_search("foo")
    assert output == [Document(page_content=initial_content, metadata={"page": "0"})]


def test_clarifai_with_metadatas() -> None:
    """Test end to end construction and search with metadata."""
    texts = ["oof", "rab", "zab"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    USER_ID = "minhajul"
    APP_ID = "test-lang-2"
    NUMBER_OF_DOCS = 1
    docsearch = Clarifai.from_texts(
        user_id=USER_ID,
        app_id=APP_ID,
        texts=texts,
        pat=None,
        number_of_docs=NUMBER_OF_DOCS,
        metadatas=metadatas,
    )
    time.sleep(2.5)
    output = docsearch.similarity_search("oof", k=1)
    assert output == [Document(page_content="oof", metadata={"page": "0"})]


def test_clarifai_with_metadatas_with_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["oof", "rab", "zab"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    USER_ID = "minhajul"
    APP_ID = "test-lang-2"
    NUMBER_OF_DOCS = 1
    docsearch = Clarifai.from_texts(
        user_id=USER_ID,
        app_id=APP_ID,
        texts=texts,
        pat=None,
        number_of_docs=NUMBER_OF_DOCS,
        metadatas=metadatas,
    )
    time.sleep(2.5)
    output = docsearch.similarity_search_with_score("oof", k=1)
    assert output[0][0] == Document(page_content="oof", metadata={"page": "0"})
    assert abs(output[0][1] - 1.0) < 0.001
