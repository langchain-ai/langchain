import unittest
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import requests

from langchain.docstore.document import Document
from langchain.document_loaders.confluence import ConfluenceLoader, ContentFormat


@pytest.fixture
def mock_confluence():  # type: ignore
    with patch("atlassian.Confluence") as mock_confluence:
        yield mock_confluence


@pytest.mark.requires("atlassian", "bs4", "lxml")
class TestConfluenceLoader:
    CONFLUENCE_URL = "https://example.atlassian.com/wiki"
    MOCK_USERNAME = "user@gmail.com"
    MOCK_API_TOKEN = "api_token"
    MOCK_SPACE_KEY = "spaceId123"

    def test_confluence_loader_initialization(self, mock_confluence: MagicMock) -> None:
        ConfluenceLoader(
            self.CONFLUENCE_URL,
            username=self.MOCK_USERNAME,
            api_key=self.MOCK_API_TOKEN,
        )
        mock_confluence.assert_called_once_with(
            url=self.CONFLUENCE_URL,
            username="user@gmail.com",
            password="api_token",
            cloud=True,
        )

    def test_confluence_loader_initialization_invalid(self) -> None:
        with pytest.raises(ValueError):
            ConfluenceLoader(
                self.CONFLUENCE_URL,
                username=self.MOCK_USERNAME,
                api_key=self.MOCK_API_TOKEN,
                token="foo",
            )

        with pytest.raises(ValueError):
            ConfluenceLoader(
                self.CONFLUENCE_URL,
                username=self.MOCK_USERNAME,
                api_key=self.MOCK_API_TOKEN,
                oauth2={
                    "access_token": "bar",
                    "access_token_secret": "bar",
                    "consumer_key": "bar",
                    "key_cert": "bar",
                },
            )

        with pytest.raises(ValueError):
            ConfluenceLoader(
                self.CONFLUENCE_URL,
                username=self.MOCK_USERNAME,
                api_key=self.MOCK_API_TOKEN,
                session=requests.Session(),
            )

    def test_confluence_loader_initialization_from_env(
        self, mock_confluence: MagicMock
    ) -> None:
        with unittest.mock.patch.dict(
            "os.environ",
            {
                "CONFLUENCE_USERNAME": self.MOCK_USERNAME,
                "CONFLUENCE_API_TOKEN": self.MOCK_API_TOKEN,
            },
        ):
            ConfluenceLoader(url=self.CONFLUENCE_URL)
            mock_confluence.assert_called_with(
                url=self.CONFLUENCE_URL, username=None, password=None, cloud=True
            )

    def test_confluence_loader_load_data_invalid_args(self) -> None:
        confluence_loader = ConfluenceLoader(
            self.CONFLUENCE_URL,
            username=self.MOCK_USERNAME,
            api_key=self.MOCK_API_TOKEN,
        )

        with pytest.raises(
            ValueError,
            match="Must specify at least one among `space_key`, `page_ids`, `label`, `cql` parameters.",  # noqa: E501
        ):
            confluence_loader.load()

    def test_confluence_loader_load_data_by_page_ids(
        self, mock_confluence: MagicMock
    ) -> None:
        mock_confluence.get_page_by_id.side_effect = [
            self._get_mock_page("123"),
            self._get_mock_page("456"),
        ]
        mock_confluence.get_all_restrictions_for_content.side_effect = [
            self._get_mock_page_restrictions("123"),
            self._get_mock_page_restrictions("456"),
        ]

        confluence_loader = self._get_mock_confluence_loader(mock_confluence)

        mock_page_ids = ["123", "456"]
        documents = confluence_loader.load(page_ids=mock_page_ids)

        assert mock_confluence.get_page_by_id.call_count == 2
        assert mock_confluence.get_all_restrictions_for_content.call_count == 2

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].page_content == "Content 123"
        assert documents[1].page_content == "Content 456"

        assert mock_confluence.get_all_pages_from_space.call_count == 0
        assert mock_confluence.get_all_pages_by_label.call_count == 0
        assert mock_confluence.cql.call_count == 0
        assert mock_confluence.get_page_child_by_type.call_count == 0

    def test_confluence_loader_load_data_by_space_id(
        self, mock_confluence: MagicMock
    ) -> None:
        # one response with two pages
        mock_confluence.get_all_pages_from_space.return_value = [
            self._get_mock_page("123"),
            self._get_mock_page("456"),
        ]
        mock_confluence.get_all_restrictions_for_content.side_effect = [
            self._get_mock_page_restrictions("123"),
            self._get_mock_page_restrictions("456"),
        ]

        confluence_loader = self._get_mock_confluence_loader(mock_confluence)

        documents = confluence_loader.load(space_key=self.MOCK_SPACE_KEY, max_pages=2)

        assert mock_confluence.get_all_pages_from_space.call_count == 1

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].page_content == "Content 123"
        assert documents[1].page_content == "Content 456"

        assert mock_confluence.get_page_by_id.call_count == 0
        assert mock_confluence.get_all_pages_by_label.call_count == 0
        assert mock_confluence.cql.call_count == 0
        assert mock_confluence.get_page_child_by_type.call_count == 0

    def test_confluence_loader_when_content_format_and_keep_markdown_format_enabled(
        self, mock_confluence: MagicMock
    ) -> None:
        # one response with two pages
        mock_confluence.get_all_pages_from_space.return_value = [
            self._get_mock_page("123", ContentFormat.VIEW),
            self._get_mock_page("456", ContentFormat.VIEW),
        ]
        mock_confluence.get_all_restrictions_for_content.side_effect = [
            self._get_mock_page_restrictions("123"),
            self._get_mock_page_restrictions("456"),
        ]

        confluence_loader = self._get_mock_confluence_loader(mock_confluence)

        documents = confluence_loader.load(
            space_key=self.MOCK_SPACE_KEY,
            content_format=ContentFormat.VIEW,
            keep_markdown_format=True,
            max_pages=2,
        )

        assert mock_confluence.get_all_pages_from_space.call_count == 1

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].page_content == "Content 123\n\n"
        assert documents[1].page_content == "Content 456\n\n"

        assert mock_confluence.get_page_by_id.call_count == 0
        assert mock_confluence.get_all_pages_by_label.call_count == 0
        assert mock_confluence.cql.call_count == 0
        assert mock_confluence.get_page_child_by_type.call_count == 0

    def _get_mock_confluence_loader(
        self, mock_confluence: MagicMock
    ) -> ConfluenceLoader:
        confluence_loader = ConfluenceLoader(
            self.CONFLUENCE_URL,
            username=self.MOCK_USERNAME,
            api_key=self.MOCK_API_TOKEN,
        )
        confluence_loader.confluence = mock_confluence
        return confluence_loader

    def _get_mock_page(
        self, page_id: str, content_format: ContentFormat = ContentFormat.STORAGE
    ) -> Dict:
        return {
            "id": f"{page_id}",
            "title": f"Page {page_id}",
            "body": {
                f"{content_format.name.lower()}": {"value": f"<p>Content {page_id}</p>"}
            },
            "status": "current",
            "type": "page",
            "_links": {
                "self": f"{self.CONFLUENCE_URL}/rest/api/content/{page_id}",
                "tinyui": "/x/tiny_ui_link",
                "editui": f"/pages/resumedraft.action?draftId={page_id}",
                "webui": f"/spaces/{self.MOCK_SPACE_KEY}/overview",
            },
        }

    def _get_mock_page_restrictions(self, page_id: str) -> Dict:
        return {
            "read": {
                "operation": "read",
                "restrictions": {
                    "user": {"results": [], "start": 0, "limit": 200, "size": 0},
                    "group": {"results": [], "start": 0, "limit": 200, "size": 0},
                },
                "_expandable": {"content": f"/rest/api/content/{page_id}"},
                "_links": {
                    "self": f"{self.CONFLUENCE_URL}/rest/api/content/{page_id}/restriction/byOperation/read"  # noqa: E501
                },
            },
            "update": {
                "operation": "update",
                "restrictions": {
                    "user": {"results": [], "start": 0, "limit": 200, "size": 0},
                    "group": {"results": [], "start": 0, "limit": 200, "size": 0},
                },
                "_expandable": {"content": f"/rest/api/content/{page_id}"},
                "_links": {
                    "self": f"{self.CONFLUENCE_URL}/rest/api/content/{page_id}/restriction/byOperation/update"  # noqa: E501
                },
            },
            "_links": {
                "self": f"{self.CONFLUENCE_URL}/rest/api/content/{page_id}/restriction/byOperation",  # noqa: E501
                "base": self.CONFLUENCE_URL,
                "context": "/wiki",
            },
        }
