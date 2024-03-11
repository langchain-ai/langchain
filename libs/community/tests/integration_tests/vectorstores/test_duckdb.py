import pytest
import duckdb
import os
from langchain_community.vectorstores import DuckDB
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

@pytest.fixture
def duckdb_connection():
    import duckdb
    # Setup a temporary DuckDB database
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()

@pytest.fixture
def embeddings():
    return FakeEmbeddings()

@pytest.fixture
def texts():
    return ["text 1", "text 2", "item 3"]

@pytest.mark.requires("duckdb")
def test_duckdb_with_connection(duckdb_connection, embeddings, texts):
    store = DuckDB(connection=duckdb_connection, embedding=embeddings, table_name="test_table")
    store.add_texts(texts)
    result = store.similarity_search("text 1")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts

@pytest.mark.requires("duckdb")
def test_duckdb_without_connection(embeddings, texts):
    store = DuckDB(embedding=embeddings, table_name="test_table")
    store.add_texts(texts)
    result = store.similarity_search("text 1")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts

@pytest.mark.requires("duckdb")
def test_duckdb_add_texts(embeddings):
    store = DuckDB(embedding=embeddings, table_name="test_table")
    store.add_texts(["text 2"])
    result = store.similarity_search("text 2")
    result_texts = [doc.page_content for doc in result]
    assert "text 2" in result_texts
