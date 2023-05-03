import unittest
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from langchain.docstore.document import Document
from langchain.document_loaders.confluence import ConfluenceLoader


@pytest.fixture
def mock_confluence() -> Generator[MagicMock, None, None]:
    with patch("atlassian.Confluence") as mock_confluence:
        yield mock_confluence


CONFLUENCE_URL = "https://example.atlassian.com/wiki"
MOCK_USERNAME = "user@gmail.com"
MOCK_API_TOKEN = "api_token"


class TestConfluenceLoader:
    def test_confluence_loader_initialization(self, mock_confluence: MagicMock) -> None:
        ConfluenceLoader(
            url=CONFLUENCE_URL, username=MOCK_USERNAME, api_key=MOCK_API_TOKEN
        )
        mock_confluence.assert_called_once_with(
            url=CONFLUENCE_URL,
            username="user@gmail.com",
            password="api_token",
            cloud=True,
        )

    def test_confluence_loader_initialization_from_env(
        self, mock_confluence: MagicMock
    ) -> None:
        with unittest.mock.patch.dict(
            "os.environ",
            {
                "CONFLUENCE_USERNAME": MOCK_USERNAME,
                "CONFLUENCE_API_TOKEN": MOCK_API_TOKEN,
            },
        ):
            ConfluenceLoader(url=CONFLUENCE_URL)
            mock_confluence.assert_called_with(
                url=CONFLUENCE_URL, username=None, password=None, cloud=True
            )

    def test_confluence_loader_load_data_invalid_args(self) -> None:
        confluence_loader = ConfluenceLoader(
            url=CONFLUENCE_URL, username=MOCK_USERNAME, api_key=MOCK_API_TOKEN
        )

        with pytest.raises(
            ValueError,
            match="Must specify at least one among `space_key`, `page_ids`,`label`, `cql` parameters.",
        ):
            confluence_loader.load()

    def test_confluence_loader_load_data_by_page_ids(
        self, mock_confluence: MagicMock
    ) -> None:
        mock_confluence.get_page_by_id.side_effect = [
            {
                "id": "123",
                "title": "Page 123",
                "body": {"storage": {"value": "<p>Content 123</p>"}},
            },
            {
                "id": "456",
                "title": "Page 456",
                "body": {"storage": {"value": "<p>Content 456</p>"}},
            },
        ]

        confluence_loader = ConfluenceLoader(
            url=CONFLUENCE_URL, username=MOCK_USERNAME, api_key=MOCK_API_TOKEN
        )
        confluence_loader.confluence = mock_confluence

        mock_page_ids = ["123", "456"]
        documents = confluence_loader.load(page_ids=mock_page_ids)

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].page_content == "Content 123\n\n"
        assert documents[1].page_content == "\n\nContent 456\n\n"

        assert mock_confluence.get_page_by_id.call_count == 2

        assert mock_confluence.get_all_pages_from_space.call_count == 0
        assert mock_confluence.get_all_pages_by_label.call_count == 0
        assert mock_confluence.cql.call_count == 0
        assert mock_confluence.get_page_child_by_type.call_count == 0

    def test_confluence_loader_load_data_by_space_id(
        self, mock_confluence: MagicMock
    ) -> None:
        # one response with two pages
        mock_confluence.get_all_pages_from_space.return_value = [
            {
                "id": "123",
                "type": "page",
                "status": "current",
                "title": "Page 123",
                "body": {"storage": {"value": "<p>Content 123</p>"}},
            },
            {
                "id": "456",
                "type": "page",
                "status": "current",
                "title": "Page 456",
                "body": {"storage": {"value": "<p>Content 456</p>"}},
            },
        ]

        confluence_loader = ConfluenceLoader(
            url=CONFLUENCE_URL, username=MOCK_USERNAME, api_key=MOCK_API_TOKEN
        )
        confluence_loader.confluence = mock_confluence

        mock_space_key = "spaceId123"
        documents = confluence_loader.load(space_key=mock_space_key)

        assert mock_confluence.get_all_pages_from_space.call_count == 1
        assert (
            mock_confluence.get_all_pages_from_space.call_args[1]["space"]
            == "spaceId123"
        )
        assert mock_confluence.get_all_pages_from_space.call_args[1]["start"] == 0
        assert (
            mock_confluence.get_all_pages_from_space.call_args[1]["expand"]
            == "body.storage.value"
        )

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].page_content == "Content 123\n\n"
        assert documents[1].page_content == "\n\nContent 456\n\n"

        assert mock_confluence.get_page_by_id.call_count == 0
        assert mock_confluence.get_all_pages_by_label.call_count == 0
        assert mock_confluence.cql.call_count == 0
        assert mock_confluence.get_page_child_by_type.call_count == 0
