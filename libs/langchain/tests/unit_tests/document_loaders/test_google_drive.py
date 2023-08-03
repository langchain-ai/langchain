import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from langchain.document_loaders.google_drive import GoogleDriveLoader
from tests.unit_tests.llms.fake_llm import FakeLLM
from tests.unit_tests.utilities.test_google_drive import (
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
def test_load_returns_list_of_google_documents_single(
    google_workspace: MagicMock,
) -> None:
    loader = GoogleDriveLoader(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        folder_id="999",
    )
    assert loader.mode == "documents"  # Check default value
    assert loader.gsheet_mode == "single"  # Check default value
    assert loader.gslide_mode == "single"  # Check default value


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_service_account_key(google_workspace: MagicMock) -> None:
    loader = GoogleDriveLoader(
        service_account_key=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_service.json",
        template="gdrive-all-in-folder",
    )
    assert (
        loader.gdrive_api_file
        == Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_service.json"
    )


# @unittest.skipIf(not google_workspace_installed, "Google api not installed")
# def test_no_path(mocker,google_workspace) -> None:
#     import os
#     mocker.patch.dict(os.environ,{},clear=True)
#     loader = GoogleDriveLoader(
#         template="gdrive-all-in-folder",
#     )
#     assert loader.gdrive_api_file == Path.home() / ".credentials" / "keys.json"


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_credentials_path(mocker: MockerFixture, google_workspace: MagicMock) -> None:
    loader = GoogleDriveLoader(
        credentials_path=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        template="gdrive-all-in-folder",
    )
    assert (
        loader.gdrive_api_file
        == Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json"
    )


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_folder_id(google_workspace: MagicMock) -> None:
    loader = GoogleDriveLoader(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        folder_id="999",
    )
    docs = loader.load()
    assert len(docs) == 3


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_query(google_workspace: MagicMock) -> None:
    loader = GoogleDriveLoader(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        query="",
        template="gdrive-query",
    )
    docs = loader.load()
    assert len(docs) == 3


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_document_ids(google_workspace: MagicMock) -> None:
    loader = GoogleDriveLoader(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        document_ids=["1", "1"],
    )
    docs = loader.load()
    assert len(docs) == 2


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_files_ids(google_workspace: MagicMock) -> None:
    loader = GoogleDriveLoader(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        file_ids=["1", "2"],
    )
    docs = loader.load()
    assert len(docs) == 2


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_update_description_with_summary(google_workspace: MagicMock) -> None:
    loader = GoogleDriveLoader(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        file_ids=["1", "2"],
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    result = list(
        loader.lazy_update_description_with_summary(
            llm=FakeLLM(), force=True, prompt=None, verbose=True, query=""
        )
    )
    assert len(result) == 2

    result = list(
        loader.lazy_update_description_with_summary(
            llm=FakeLLM(), force=False, prompt=None, query=""
        )
    )
    assert len(result) == 0
