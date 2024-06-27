"""Test AzureSQLServer_VectorStore functionality."""

import os

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.azure_sql_server import AzureSQLServer_VectorStore

_CONNECTION_STRING = os.environ.get("TEST_AZURESQLSERVER_CONNECTION_STRING")


def test_azure_sql_server_add_text() -> None:
    """Test that add text returns equivalent number of ids of input texts."""
    texts = ["rabbit", "cherry", "hamster", "cat", "elderberry"]
    metadatas = [
        {"color": "black", "type": "pet", "length": 6},
        {"color": "red", "type": "fruit", "length": 6},
        {"color": "brown", "type": "pet", "length": 7},
        {"color": "black", "type": "pet", "length": 3},
        {"color": "blue", "type": "fruit", "length": 10},
    ]
    store = AzureSQLServer_VectorStore(
        connection_string=_CONNECTION_STRING,
        embedding_function=FakeEmbeddings(size=1536),
        table_name="langchain_vector_store_tests",
    )
    output = store.add_texts(texts, metadatas)
    assert len(output) == len(texts)
