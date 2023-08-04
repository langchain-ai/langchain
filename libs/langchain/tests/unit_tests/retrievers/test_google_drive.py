import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from langchain.retrievers.google_drive import GoogleDriveRetriever
from tests.unit_tests.utilities.test_google_drive import (
    _text_text,
    gdrive_docs,
    google_workspace_installed,
    patch_google_workspace,
)


@pytest.fixture
def google_workspace(mocker: MockerFixture) -> MagicMock:
    return patch_google_workspace(
        mocker, [{"nextPageToken": None, "files": gdrive_docs}]
    )


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_get_relevant_documents(
    mocker: MockerFixture,
) -> None:
    patch_google_workspace(mocker, [{"nextPageToken": None, "files": [_text_text]}])
    retriever = GoogleDriveRetriever(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
    )
    docs = retriever.get_relevant_documents("machine learning")
    assert len(docs) == 1


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_extra_parameters(
    mocker: MockerFixture,
) -> None:
    patch_google_workspace(mocker, [{"nextPageToken": None, "files": [_text_text]}])
    retriever = GoogleDriveRetriever(
        template="gdrive-mime-type-in-folder",
        folder_id="root",
        mime_type="application/vnd.google-apps.document",  # Only Google Docs
        num_results=2,
        mode="snippets",
        includeItemsFromAllDrives=False,
        supportsAllDrives=False,
    )
    retriever.get_relevant_documents("machine learning")
