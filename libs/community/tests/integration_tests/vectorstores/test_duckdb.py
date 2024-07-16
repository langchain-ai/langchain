from typing import Dict, Iterator, List
from uuid import uuid4

import duckdb
import pytest

from langchain_community.vectorstores import DuckDB
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.fixture
def duckdb_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    # Setup a temporary DuckDB database
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def texts() -> List[str]:
    return ["text 1", "text 2", "item 3"]


@pytest.fixture
def metadatas() -> List[Dict[str, str]]:
    return [
        {"source": "Document 1"},
        {"source": "Document 2"},
        {"source": "Document 3"},
    ]


@pytest.mark.requires("duckdb")
def test_duckdb_with_connection(
    duckdb_connection: duckdb.DuckDBPyConnection,
    embeddings: FakeEmbeddings,
    texts: List[str],
) -> None:
    store = DuckDB(
        connection=duckdb_connection, embedding=embeddings, table_name="test_table"
    )
    store.add_texts(texts)
    result = store.similarity_search("text 1")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts


@pytest.mark.requires("duckdb")
def test_duckdb_without_connection(
    embeddings: FakeEmbeddings, texts: List[str]
) -> None:
    store = DuckDB(embedding=embeddings, table_name="test_table")
    store.add_texts(texts)
    result = store.similarity_search("text 1")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts


@pytest.mark.requires("duckdb")
def test_duckdb_add_texts(embeddings: FakeEmbeddings) -> None:
    store = DuckDB(embedding=embeddings, table_name="test_table")
    store.add_texts(["text 2"])
    result = store.similarity_search("text 2")
    result_texts = [doc.page_content for doc in result]
    assert "text 2" in result_texts


@pytest.mark.requires("duckdb")
def test_duckdb_add_texts_with_metadata(
    duckdb_connection: duckdb.DuckDBPyConnection, embeddings: FakeEmbeddings
) -> None:
    store = DuckDB(
        connection=duckdb_connection,
        embedding=embeddings,
        table_name="test_table_with_metadata",
    )
    texts = ["text with metadata 1", "text with metadata 2"]
    metadatas = [
        {"author": "Author 1", "date": "2021-01-01"},
        {"author": "Author 2", "date": "2021-02-01"},
    ]

    # Add texts along with their metadata
    store.add_texts(texts, metadatas=metadatas)

    # Perform a similarity search to retrieve the documents
    result = store.similarity_search("text with metadata", k=2)

    # Check if the metadata is correctly associated with the texts
    assert len(result) == 2, "Should return two results"
    assert (
        result[0].metadata.get("author") == "Author 1"
    ), "Metadata for Author 1 should be correctly retrieved"
    assert (
        result[0].metadata.get("date") == "2021-01-01"
    ), "Date for Author 1 should be correctly retrieved"
    assert (
        result[1].metadata.get("author") == "Author 2"
    ), "Metadata for Author 2 should be correctly retrieved"
    assert (
        result[1].metadata.get("date") == "2021-02-01"
    ), "Date for Author 2 should be correctly retrieved"


@pytest.mark.requires("duckdb")
def test_duckdb_add_texts_with_predefined_ids(
    duckdb_connection: duckdb.DuckDBPyConnection, embeddings: FakeEmbeddings
) -> None:
    store = DuckDB(
        connection=duckdb_connection,
        embedding=embeddings,
        table_name="test_table_predefined_ids",
    )
    texts = ["unique text 1", "unique text 2"]
    predefined_ids = [str(uuid4()), str(uuid4())]  # Generate unique IDs

    # Add texts with the predefined IDs
    store.add_texts(texts, ids=predefined_ids)

    # Perform a similarity search for each text and check if it's found
    for text in texts:
        result = store.similarity_search(text)

        found_texts = [doc.page_content for doc in result]
        assert (
            text in found_texts
        ), f"Text '{text}' was not found in the search results."


@pytest.mark.requires("duckdb")
def test_duckdb_from_texts(
    duckdb_connection: duckdb.DuckDBPyConnection,
    embeddings: FakeEmbeddings,
    texts: List[str],
    metadatas: List[Dict[str, str]],
) -> None:
    # Initialize DuckDB from texts using the from_texts class method
    store = DuckDB.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        connection=duckdb_connection,
        table_name="test_from_texts_table",
    )

    # Perform a similarity search to retrieve the documents
    query_text = "sample text"
    result = store.similarity_search(query_text, k=2)

    # Verify that the vector store was populated and can return results
    assert len(result) > 0, "Should return at least one result"

    # Optionally, check that metadata is correctly associated with the texts
    for doc in result:
        assert "source" in doc.metadata, "Document metadata should include 'source' key"
