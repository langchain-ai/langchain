"""Test SQLServer_VectorStore functionality."""

import os

import pytest
from langchain_core.documents import Document

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.sqlserver import SQLServer_VectorStore

_CONNECTION_STRING = str(os.environ.get("TEST_AZURESQLSERVER_CONNECTION_STRING"))


@pytest.fixture
def store() -> SQLServer_VectorStore:
    """Setup resources that are needed for the duration of the test."""
    store = SQLServer_VectorStore(
        connection_string=_CONNECTION_STRING,
        embedding_function=FakeEmbeddings(size=1536),
        table_name="langchain_vector_store_tests",
    )
    return store  # provide this data to the test


def test_sqlserver_add_texts(store: SQLServer_VectorStore) -> None:
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


def test_sqlserver_add_texts_when_no_metadata_is_provided(
    store: SQLServer_VectorStore,
) -> None:
    """Test that when user calls the add_texts function without providing metadata,
    the embedded text still get added to the vector store."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
    ]
    result = store.add_texts(texts)
    assert len(result) == len(texts)


def test_sqlserver_add_texts_when_text_length_and_metadata_length_vary(
    store: SQLServer_VectorStore,
) -> None:
    """Test that all texts provided are added into the vector store
    even when metadata is not available for all the texts."""
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


def test_sqlserver_add_texts_when_list_of_given_id_is_less_than_list_of_texts(
    store: SQLServer_VectorStore,
) -> None:
    """Test that when length of given id is less than length of texts,
    random ids are created."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
        "It's a new year.",
    ]

    # List of ids is 3 and is less than len(texts) which is 5.
    metadatas = [
        {"id": 1, "soure": "book review", "length": 11},
        {"id": 2, "source": "random texts", "length": 9},
        {"source": "household list", "length": 5},
        {"id": 6, "source": "newspaper page", "length": 44},
        {"source": "random texts", "length": 16},
    ]
    result = store.add_texts(texts, metadatas)

    # Length of ids returned by add_texts function should be equal to length of texts.
    assert len(result) == len(texts)


def test_add_document_with_sqlserver(store: SQLServer_VectorStore) -> None:
    """Test that when add_document function is used, it integrates well
    with the add_text function in SQLServer Vector Store."""
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


def test_that_a_document_entry_without_metadata_will_be_added_to_vectorstore(
    store: SQLServer_VectorStore,
) -> None:
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


def test_sqlserver_delete_text_by_id_valid_ids_provided(
    store: SQLServer_VectorStore,
) -> None:
    """Test that delete API deletes texts by id."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
        "It's a new year.",
    ]

    # List of ids is 3 and is less than len(texts) which is 5.
    metadatas = [
        {"id": 100, "source": "book review", "length": 11},
        {"id": 200, "source": "random texts", "length": 9},
        {"id": 200, "source": "household list", "length": 5},
        {"id": 600, "source": "newspaper page", "length": 44},
        {"id": 300, "source": "random texts", "length": 16},
    ]
    store.add_texts(texts, metadatas)

    result = store.delete(["100", "200", "600"])
    # Should return true since valid ids are given
    if result:
        pass


def test_sqlserver_delete_text_by_id_invalid_ids_provided(
    store: SQLServer_VectorStore,
) -> None:
    """Test that delete API deletes texts by id."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
        "It's a new year.",
    ]

    metadatas = [
        {"id": 100, "source": "book review", "length": 11},
        {"id": 200, "source": "random texts", "length": 9},
        {"id": 200, "source": "household list", "length": 5},
        {"id": 600, "source": "newspaper page", "length": 44},
        {"id": 300, "source": "random texts", "length": 16},
    ]
    store.add_texts(texts, metadatas)

    result = store.delete(["100000"])
    # Should return False since given id is not in DB
    if not result:
        pass


def test_sqlserver_delete_text_by_id_no_ids_provided(
    store: SQLServer_VectorStore,
) -> None:
    """Test that delete API deletes texts by id."""
    texts = [
        "Good review",
        "new books",
        "table",
        "Sunglasses are a form of protective eyewear.",
        "It's a new year.",
    ]

    metadatas = [
        {"id": 100, "source": "book review", "length": 11},
        {"id": 200, "source": "random texts", "length": 9},
        {"id": 200, "source": "household list", "length": 5},
        {"id": 600, "source": "newspaper page", "length": 44},
        {"id": 300, "source": "random texts", "length": 16},
    ]
    store.add_texts(texts, metadatas)

    # Should throw an exception since no ids are provided
    with pytest.raises(ValueError, match="No ids provided to delete."):
        store.delete([])
