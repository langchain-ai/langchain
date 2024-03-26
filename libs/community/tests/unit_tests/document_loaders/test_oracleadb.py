from typing import Dict, List
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from langchain_community.document_loaders.oracleadb_loader import (
    OracleAutonomousDatabaseLoader,
)


def raw_docs() -> List[Dict]:
    return [
        {"FIELD1": "1", "FIELD_JSON": {"INNER_FIELD1": "1", "INNER_FIELD2": "1"}},
        {"FIELD1": "2", "FIELD_JSON": {"INNER_FIELD1": "2", "INNER_FIELD2": "2"}},
        {"FIELD1": "3", "FIELD_JSON": {"INNER_FIELD1": "3", "INNER_FIELD2": "3"}},
    ]


def expected_documents() -> List[Document]:
    return [
        Document(
            page_content="{'FIELD1': '1', 'FIELD_JSON': "
            "{'INNER_FIELD1': '1', 'INNER_FIELD2': '1'}}",
            metadata={"FIELD1": "1"},
        ),
        Document(
            page_content="{'FIELD1': '2', 'FIELD_JSON': "
            "{'INNER_FIELD1': '2', 'INNER_FIELD2': '2'}}",
            metadata={"FIELD1": "2"},
        ),
        Document(
            page_content="{'FIELD1': '3', 'FIELD_JSON': "
            "{'INNER_FIELD1': '3', 'INNER_FIELD2': '3'}}",
            metadata={"FIELD1": "3"},
        ),
    ]


@patch(
    "langchain_community.document_loaders.oracleadb_loader.OracleAutonomousDatabaseLoader._run_query"
)
def test_oracle_loader_load(mock_query: MagicMock) -> None:
    """Test oracleDB loader load function."""

    mock_query.return_value = raw_docs()
    loader = OracleAutonomousDatabaseLoader(
        query="Test query",
        user="Test user",
        password="Test password",
        connection_string="Test connection string",
        metadata=["FIELD1"],
    )

    documents = loader.load()

    assert documents == expected_documents()
