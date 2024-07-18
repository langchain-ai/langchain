"""Test SQLServer_VectorStore functionality."""

import os

import pytest
from langchain_core.documents import Document

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.sqlserver import SQLServer_VectorStore

_CONNECTION_STRING = os.environ.get("TEST_AZURESQLSERVER_CONNECTION_STRING")


@pytest.fixture
def store():
    """Setup resources that are needed for the duration of the test."""
    store = SQLServer_VectorStore(
        connection_string=_CONNECTION_STRING,
        embedding_function=FakeEmbeddings(size=1536),
        table_name="langchain_vector_store_tests",
    )
    yield store  # provide this data to the test


def test_sqlserver_add_texts(store) -> None:
    """Test that add text returns equivalent number of ids of input texts."""
    texts = ["rabbit", "cherry", "hamster", "cat", "elderberry"]
    metadatas = [
        {"color": "black", "type": "pet", "length": 6},
        {"color": "red", "type": "fruit", "length": 6},
        {"color": "brown", "type": "pet", "length": 7},
        {"color": "black", "type": "pet", "length": 3},
        {"color": "blue", "type": "fruit", "length": 10},
    ]
    result = store.add_texts(texts, metadatas)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_no_metadata_is_provided(store) -> None:
    """Test that when user calls the add_texts function without providing metadata, the embedded text still get added to the vector store."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
    ]
    result = store.add_texts(texts)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_text_length_and_metadata_length_vary(store) -> None:
    """Test that all texts provided are added into the vector store even when metadata is not available for all the texts."""
    # The text 'elderberry' and its embedded value should be added to the vector store.
    texts = ["rabbit", "cherry", "hamster", "cat", "elderberry"]
    metadatas = [
        {"color": "black", "type": "pet", "length": 6},
        {"color": "red", "type": "fruit", "length": 6},
        {"color": "brown", "type": "pet", "length": 7},
        {"color": "black", "type": "pet", "length": 3},
    ]
    result = store.add_texts(texts, metadatas)
    assert len(result) == len(texts)


def test_add_document_with_sqlserver(store) -> None:
    """Test that when add_document function is used, it integerates well with the add_text function in SQLServer Vector Store."""
    docs = [
        Document(
            page_content="rabbit",
            metadata={"color": "black", "type": "pet", "length": 6},
        ),
        Document(
            page_content="cherry",
            metadata={"color": "red", "type": "fruit", "length": 6},
        ),
        Document(
            page_content="hamster",
            metadata={"color": "brown", "type": "pet", "length": 7},
        ),
        Document(
            page_content="cat", metadata={"color": "black", "type": "pet", "length": 3}
        ),
        Document(
            page_content="elderberry",
            metadata={"color": "blue", "type": "fruit", "length": 10},
        ),
    ]
    result = store.add_documents(docs)
    assert len(result) == len(docs)


def test_that_a_document_entry_without_metadata_will_be_added_to_vectorstore(store):
    docs = [
        Document(
            page_content="rabbit",
            metadata={"color": "black", "type": "pet", "length": 6},
        ),
        Document(
            page_content="cherry",
        ),
        Document(
            page_content="hamster",
            metadata={"color": "brown", "type": "pet", "length": 7},
        ),
        Document(page_content="cat"),
        Document(
            page_content="elderberry",
            metadata={"color": "blue", "type": "fruit", "length": 10},
        ),
    ]
    result = store.add_documents(docs)
    assert len(result) == len(docs)
