from typing import Dict, List
from unittest.mock import MagicMock

from langchain_core.documents import Document

from langchain_community.document_loaders.oracleadb_loader import (
    AutonomousDatabaseLoader,
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


def test_oracle_loader_load() -> None:
    """Test oracleDB loader load function."""

    loader = AutonomousDatabaseLoader(
        query="Test query",
        user="Test user",
        password="Test password",
        connection_string="Test connection string",
        metadata=["FIELD1"],
    )

    loader._run_query = MagicMock(return_value=raw_docs())
    documents = loader.load()

    assert documents == expected_documents()
