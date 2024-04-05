from unittest.mock import Mock, patch

import pytest

from langchain_community.document_loaders.teradata import TeradataLoader


@pytest.mark.requires("teradatasql")
@patch("teradatasql.connect")
def test_query_execution_with_mock(mock_connect: Mock) -> None:

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

    assert len(documents) > 0, "No documents loaded"
    first_doc = documents[0]
    assert (
        "version" in first_doc.metadata["InfoKey"].lower()
    ), "InfoKey 'version' not found in metadata"

    assert first_doc.page_content is not None, "InfoData is None"
