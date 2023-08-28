import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

#from langchain_googledrive.document_loaders.google_drive import GoogleDriveLoader
from langchain.document_loaders.google_drive import GoogleDriveLoader

from tests.unit_tests.llms.fake_llm import FakeLLM
from tests.unit_tests.utilities.test_google_drive import (
    gdrive_docs,
    google_workspace_installed,
    patch_google_workspace,
)

try:
    import unstructured
    unstructured_installed= True
except ImportError:
    unstructured_installed= False


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


# @unittest.skipIf(not google_workspace_installed, "Google api not installed")
# def test_no_path(mocker,google_workspace) -> None:
#     import os
#     mocker.patch.dict(os.environ,{},clear=True)
#     loader = GoogleDriveLoader(
#         template="gdrive-all-in-folder",
#     )
#     assert loader.gdrive_api_file == Path.home() / ".credentials" / "keys.json"


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
        gdrive_api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        query="",
        template="gdrive-query",
    )
    docs = loader.load()
    assert len(docs) == 3


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


## --------- Test deprecated API --------------


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_deprecated_document_ids(google_workspace: MagicMock) -> None:
    with pytest.deprecated_call() as w:
        loader = GoogleDriveLoader(
            # api_file=Path(__file__).parent.parent
            # / "utilities"
            # / "examples"
            # / "gdrive_credentials.json",
            document_ids=["1", "1"],
        )
        docs = loader.load()
        assert len(docs) == 2
    assert [str(warn.message) for warn in w.list] == [
        "document_ids and file_ids are deprecated. Use templates."
    ]


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_deprecated_files_ids(google_workspace: MagicMock) -> None:
    with pytest.deprecated_call() as w:
        loader = GoogleDriveLoader(
            # api_file=Path(__file__).parent.parent
            # / "utilities"
            # / "examples"
            # / "gdrive_credentials.json",
            file_ids=["1", "2"],
        )
        docs = loader.load()
        assert len(docs) == 2
    assert [str(warn.message) for warn in w.list] == [
        "document_ids and file_ids are deprecated. Use templates."
    ]


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_deprecated_file_loader_cls(google_workspace: MagicMock) -> None:
    from langchain.document_loaders import UnstructuredFileIOLoader

    with pytest.deprecated_call() as w:
        folder_id = "root"
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            file_loader_cls=UnstructuredFileIOLoader,
            file_loader_kwargs={"mode": "elements"},
        )
        docs = loader.load()
        assert len(docs) == 3
    assert [str(warn.message) for warn in w.list] == [
        "file_loader_cls and file_loader_kwargs are deprecated. Use conv_mapping.",
    ]


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_deprecated_files_ids_and_load_trashed_files(
    google_workspace: MagicMock,
) -> None:
    with pytest.deprecated_call() as w:
        loader = GoogleDriveLoader(
            folder_id="999",
            load_trashed_files=True,
        )
        docs = loader.load()
        assert len(docs) == 3
    assert [str(warn.message) for warn in w.list] == [
        "load_trashed_files is deprecated. Use a template.",
    ]


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_deprecated_file_types(google_workspace: MagicMock) -> None:
    with pytest.deprecated_call() as w:
        loader = GoogleDriveLoader(
            folder_id="root",
            file_types=[
                "document",
                "sheet",
                # "pdf",
                "application/vnd.google-apps.document",
                "application/vnd.google-apps.presentation",
                "application/vnd.google-apps.spreadsheet",
                # "application/pdf",
            ],
        )
        docs = loader.load()
        assert len(docs) == 3
    assert [str(warn.message) for warn in w.list] == [
        "file_types are deprecated. Use conv_mapping."
    ]


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_deprecated_service_account_key(google_workspace: MagicMock) -> None:
    with pytest.deprecated_call() as w:
        file_id = "1"
        loader = GoogleDriveLoader(
            service_account_key=Path(__file__).parent.parent
            / "utilities"
            / "examples"
            / "gdrive_service.json",
            file_ids=[file_id],
        )
        assert (
            loader.gdrive_api_file
            == Path(__file__).parent.parent
            / "utilities"
            / "examples"
            / "gdrive_service.json"
        )
        docs = loader.load()
        assert len(docs) == 1
    assert [str(warn.message) for warn in w.list] == [
        "document_ids and file_ids are deprecated. Use templates.",
        "service_account_key was deprecated. Use GOOGLE_ACCOUNT_FILE env. variable.",
    ]


# Test older ipynb script
@unittest.skipIf(not google_workspace_installed, "Google api not installed")
@unittest.skipIf(not unstructured_installed, "Unstructured api not installed")
def test_old_ipynb(google_workspace: MagicMock) -> None:
    # Step 1
    loader = GoogleDriveLoader(
        folder_id="999",
        recursive=False,
    )
    docs = loader.load()

    # Step 2
    loader = GoogleDriveLoader(
        folder_id="999", file_types=["document", "sheet"], recursive=False
    )
    loader.load()

    # Step 3
    from langchain.document_loaders import UnstructuredFileIOLoader

    file_id = "1"
    loader = GoogleDriveLoader(
        file_ids=[file_id],
        file_loader_cls=UnstructuredFileIOLoader,
        file_loader_kwargs={"mode": "elements"},
    )
    docs = loader.load()

    # Step 4
    folder_id = "999"
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        file_loader_cls=UnstructuredFileIOLoader,
        file_loader_kwargs={"mode": "elements"},
    )
    docs = loader.load()
    print(docs)
