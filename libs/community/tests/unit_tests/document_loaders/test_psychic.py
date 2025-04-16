from __future__ import annotations

from typing import TYPE_CHECKING, Dict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders.psychic import PsychicLoader

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def mock_psychic() -> Iterator[MagicMock]:
    with patch("psychicapi.Psychic") as mock_psychic:
        yield mock_psychic


@pytest.fixture
def mock_connector_id() -> Iterator[MagicMock]:
    with patch("psychicapi.ConnectorId") as mock_connector_id:
        yield mock_connector_id


@pytest.mark.requires("psychicapi")
class TestPsychicLoader:
    MOCK_API_KEY: str = "api_key"
    MOCK_CONNECTOR_ID: str = "notion"
    MOCK_ACCOUNT_ID: str = "account_id"

    def test_psychic_loader_initialization(
        self, mock_psychic: MagicMock, mock_connector_id: MagicMock
    ) -> None:
        PsychicLoader(
            api_key=self.MOCK_API_KEY,
            connector_id=self.MOCK_CONNECTOR_ID,
            account_id=self.MOCK_ACCOUNT_ID,
        )

        mock_psychic.assert_called_once_with(secret_key=self.MOCK_API_KEY)
        mock_connector_id.assert_called_once_with(self.MOCK_CONNECTOR_ID)

    def test_psychic_loader_load_data(self, mock_psychic: MagicMock) -> None:
        mock_get_documents_response = MagicMock()
        mock_get_documents_response.documents = [
            self._get_mock_document("123"),
            self._get_mock_document("456"),
        ]
        mock_get_documents_response.next_page_cursor = None

        mock_psychic.get_documents.return_value = mock_get_documents_response

        psychic_loader = self._get_mock_psychic_loader(mock_psychic)

        documents = psychic_loader.load()

        assert mock_psychic.get_documents.call_count == 1
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].page_content == "Content 123"
        assert documents[1].page_content == "Content 456"

    def _get_mock_psychic_loader(self, mock_psychic: MagicMock) -> PsychicLoader:
        psychic_loader = PsychicLoader(
            api_key=self.MOCK_API_KEY,
            connector_id=self.MOCK_CONNECTOR_ID,
            account_id=self.MOCK_ACCOUNT_ID,
        )
        psychic_loader.psychic = mock_psychic
        return psychic_loader

    def _get_mock_document(self, uri: str) -> Dict:
        return {"uri": f"{uri}", "title": f"Title {uri}", "content": f"Content {uri}"}
