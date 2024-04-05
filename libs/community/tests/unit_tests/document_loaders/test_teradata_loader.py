from unittest.mock import patch

import pytest

from langchain_community.document_loaders.teradata import TeradataLoader


@pytest.mark.requires("teradatasql")
@patch("teradatasql.connect")
def test_query_execution_with_mock(self, mock_connect) -> None:

    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value
    mock_cursor.description = [("InfoKey",), ("InfoData",)]
    mock_cursor.fetchall.return_value = [("version", "some_data")]

    loader = TeradataLoader(
        query="SELECT InfoKey, InfoData FROM DBC.DBCInfoTbl",
        db_url="dummy_url",
        user="demo_user",
        password="password",
        page_content_columns=["InfoData"],
        metadata_columns=["InfoKey"],
    )

    documents = list(loader.lazy_load())

    self.assertTrue(len(documents) > 0, "No documents loaded")
    first_doc = documents[0]
    self.assertIn(
        "version",
        first_doc.metadata["InfoKey"].lower(),
        "InfoKey 'version' not found in metadata",
    )
    self.assertIsNotNone(first_doc.page_content, "InfoData is None")
