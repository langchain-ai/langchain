from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders.quip import QuipLoader

try:
    from quip_api.quip import QuipClient  # noqa: F401

    quip_installed = True
except ImportError:
    quip_installed = False


@pytest.fixture
def mock_quip():  # type: ignore
    # mock quip_client
    with patch("quip_api.quip.QuipClient") as mock_quip:
        yield mock_quip


@pytest.mark.requires("quip_api")
class TestQuipLoader:
    API_URL: str = "https://example-api.quip.com"
    DOC_URL_PREFIX = ("https://example.quip.com",)
    ACCESS_TOKEN: str = "api_token"

    MOCK_FOLDER_IDS = ["ABC"]
    MOCK_THREAD_IDS = ["ABC", "DEF"]

    def test_quip_loader_initialization(self, mock_quip: MagicMock) -> None:
        QuipLoader(self.API_URL, access_token=self.ACCESS_TOKEN, request_timeout=60)
        mock_quip.assert_called_once_with(
            access_token=self.ACCESS_TOKEN, base_url=self.API_URL, request_timeout=60
        )

    def test_quip_loader_load_date_invalid_args(self) -> None:
        quip_loader = QuipLoader(
            self.API_URL, access_token=self.ACCESS_TOKEN, request_timeout=60
        )

        with pytest.raises(
            ValueError,
            match="Must specify at least one among `folder_ids`, `thread_ids` or "
            "set `include_all`_folders as True",
        ):
            quip_loader.load()

    def test_quip_loader_load_data_by_folder_id(self, mock_quip: MagicMock) -> None:
        mock_quip.get_folder.side_effect = [
            self._get_mock_folder(self.MOCK_FOLDER_IDS[0])
        ]
        mock_quip.get_thread.side_effect = [
            self._get_mock_thread(self.MOCK_THREAD_IDS[0]),
            self._get_mock_thread(self.MOCK_THREAD_IDS[1]),
        ]

        quip_loader = self._get_mock_quip_loader(mock_quip)
        documents = quip_loader.load(folder_ids=[self.MOCK_FOLDER_IDS[0]])
        assert mock_quip.get_folder.call_count == 1
        assert mock_quip.get_thread.call_count == 2
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert (
            documents[0].metadata.get("source")
            == f"https://example.quip.com/{self.MOCK_THREAD_IDS[0]}"
        )
        assert (
            documents[1].metadata.get("source")
            == f"https://example.quip.com/{self.MOCK_THREAD_IDS[1]}"
        )

    def test_quip_loader_load_data_all_folder(self, mock_quip: MagicMock) -> None:
        mock_quip.get_authenticated_user.side_effect = [
            self._get_mock_authenticated_user()
        ]

        mock_quip.get_folder.side_effect = [
            self._get_mock_folder(self.MOCK_FOLDER_IDS[0]),
        ]

        mock_quip.get_thread.side_effect = [
            self._get_mock_thread(self.MOCK_THREAD_IDS[0]),
            self._get_mock_thread(self.MOCK_THREAD_IDS[1]),
        ]

        quip_loader = self._get_mock_quip_loader(mock_quip)
        documents = quip_loader.load(include_all_folders=True)
        assert mock_quip.get_folder.call_count == 1
        assert mock_quip.get_thread.call_count == 2
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert (
            documents[0].metadata.get("source")
            == f"https://example.quip.com/{self.MOCK_THREAD_IDS[0]}"
        )
        assert (
            documents[1].metadata.get("source")
            == f"https://example.quip.com/{self.MOCK_THREAD_IDS[1]}"
        )

    def test_quip_loader_load_data_by_thread_id(self, mock_quip: MagicMock) -> None:
        mock_quip.get_thread.side_effect = [
            self._get_mock_thread(self.MOCK_THREAD_IDS[0]),
            self._get_mock_thread(self.MOCK_THREAD_IDS[1]),
        ]

        quip_loader = self._get_mock_quip_loader(mock_quip)
        documents = quip_loader.load(thread_ids=self.MOCK_THREAD_IDS)

        assert mock_quip.get_folder.call_count == 0
        assert mock_quip.get_thread.call_count == 2
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert (
            documents[0].metadata.get("source")
            == f"https://example.quip.com/{self.MOCK_THREAD_IDS[0]}"
        )
        assert (
            documents[1].metadata.get("source")
            == f"https://example.quip.com/{self.MOCK_THREAD_IDS[1]}"
        )

    def _get_mock_quip_loader(self, mock_quip: MagicMock) -> QuipLoader:
        quip_loader = QuipLoader(
            self.API_URL, access_token=self.ACCESS_TOKEN, request_timeout=60
        )
        quip_loader.quip_client = mock_quip
        return quip_loader

    def _get_mock_folder(self, folder_id: str) -> Dict:
        return {
            "folder": {
                "title": "runbook",
                "creator_id": "testing",
                "folder_type": "shared",
                "parent_id": "ABCD",
                "inherit_mode": "inherit",
                "color": "manila",
                "id": f"{folder_id}",
                "created_usec": 1668405728528904,
                "updated_usec": 1697356632672453,
                "link": "https://example.quip.com/YPH9OAR2Eu5",
            },
            "member_ids": [],
            "children": [
                {"thread_id": "ABC"},
                {"thread_id": "DEF"},
            ],
        }

    def _get_mock_thread(self, thread_id: str) -> Dict:
        return {
            "thread": {
                "author_id": "testing",
                "thread_class": "document",
                "owning_company_id": "ABC",
                "id": f"{thread_id}",
                "created_usec": 1690873126670055,
                "updated_usec": 1690874891638991,
                "title": f"Unit Test Doc {thread_id}",
                "link": f"https://example.quip.com/{thread_id}",
                "document_id": "ABC",
                "type": "document",
                "is_template": False,
                "is_deleted": False,
            },
            "user_ids": [],
            "shared_folder_ids": ["ABC"],
            "expanded_user_ids": ["ABCDEFG"],
            "invited_user_emails": [],
            "access_levels": {"ABCD": {"access_level": "OWN"}},
            "html": "<h1 id='temp:C:ABCD'>How to write Python Test </h1>",
        }

    def _get_mock_authenticated_user(self) -> Dict:
        return {"shared_folder_ids": self.MOCK_FOLDER_IDS, "id": "Test"}
