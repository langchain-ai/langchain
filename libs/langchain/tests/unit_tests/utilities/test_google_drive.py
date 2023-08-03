import io
import json
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, cast
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from langchain import PromptTemplate
from langchain.schema import Document
from langchain.utilities import GoogleDriveAPIWrapper
from langchain.utilities.google_drive import TYPE_CONV_MAPPING, GoogleDriveUtilities

try:
    from google.auth.transport.requests import Request  # noqa: F401
    from google.oauth2 import service_account  # noqa: F401
    from google.oauth2.credentials import Credentials  # noqa: F401
    from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: F401
    from googleapiclient.errors import HttpError

    google_workspace_installed = True
except ImportError:
    google_workspace_installed = False

SCOPES = [
    # See https://developers.google.com/identity/protocols/oauth2/scopes
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive",
]

mime_application_folder = "999"


@pytest.fixture
def google_workspace(mocker: MockerFixture) -> MagicMock:
    return patch_google_workspace(
        mocker, [{"nextPageToken": None, "files": gdrive_docs}]
    )


class MockCred(BaseModel):
    scopes: List[str] = SCOPES
    valid: bool = True


_application_vnd_folder = {
    "id": "998",
    "name": "folder",
    "mimeType": "application/vnd.google-apps.folder",
    "owners": [{"displayName": "John Do"}],
    "webViewLink": "https://drive.google.com/drive/folders/998",
    "modifiedTime": "2023-01-01T00:00:00.0Z",
}

_application_vnd_folder2 = {
    "id": "997",
    "name": "folder",
    "mimeType": "application/vnd.google-apps.folder",
    "owners": [{"displayName": "John Do"}],
    "webViewLink": "https://drive.google.com/drive/folders/998",
    "modifiedTime": "2023-01-01T00:00:00.0Z",
}

_application_vnd_doc = {
    "id": "1",
    "name": "vnd.google-apps.document",
    "mimeType": "application/vnd.google-apps.document",
    "owners": [{"displayName": "John Do"}],
    "webViewLink": "https://docs.google.com/document/d/1/edit?usp=drivesdk",
    "webContentLink": "https://drive.google.com/uc?id=1&export=download",
    "description": "It is a doc summary",
    "modifiedTime": "2023-01-01T00:00:00.0Z",
}
_application_vnd_slide = {
    "id": "2",
    "name": "vnd.google-apps.presentation",
    "mimeType": "application/vnd.google-apps.presentation",
    "owners": [{"displayName": "John Do"}],
    "webViewLink": "https://docs.google.com/presentation/d/2/edit?usp=drivesdk",
    "webContentLink": "https://drive.google.com/uc?id=2&export=download",
    "description": "A slide show",
    "modifiedTime": "2023-01-01T00:00:00.0Z",
}
_application_vnd_sheet = {
    "id": "3",
    "name": "vnd.google-apps.spreadsheet",
    "mimeType": "application/vnd.google-apps.spreadsheet",
    "owners": [{"displayName": "John Do"}],
    "webViewLink": "https://docs.google.com/spreadsheets/d/3/edit?usp=drivesdk",
    "webContentLink": "https://drive.google.com/uc?id=3&export=download",
    "description": "A GSheet",
    "modifiedTime": "2023-01-01T00:00:00.0Z",
}
_application_vnd_form = {
    "id": "4",
    "name": "vnd.google-apps.form",
    "mimeType": "application/vnd.google-apps.form",
    "owners": [{"displayName": "John Do"}],
    "webViewLink": "https://docs.google.com/forms/d/4/edit?usp=drivesdk",
    "webContentLink": "https://drive.google.com/uc?id=4&export=download",
    "description": "A GForm",
    "modifiedTime": "2023-01-01T00:00:00.0Z",
}
_application_vnd_jam = {
    "id": "5",
    "name": "vnd.google-apps.jam",
    "mimeType": "application/vnd.google-apps.jam",
    "owners": [{"displayName": "John Do"}],
    "webViewLink": "https://jamboard.google.com/d/5/edit",
    "webContentLink": "https://drive.google.com/uc?id=5&export=download",
    "description": "A GJam",
    "modifiedTime": "2023-01-01T00:00:00.0Z",
}
_application_vnd_drawing = {
    "id": "6",
    "name": "vnd.google-apps.drawing",
    "mimeType": "application/vnd.google-apps.drawing",
    "owners": [{"displayName": "John Do"}],
    "webViewLink": "https://docs.google.com/drawings/d/6/edit?usp=drivesdk",
    "webContentLink": "https://docs.google.com/drawings/d/6/image?w=512",
    "description": "A GDrawing",
    "modifiedTime": "2023-01-01T00:00:00.0Z",
}

_application_vnd_shortcut = {
    "id": "7",
    "name": "vnd.google-apps.shortcut.1",
    "mimeType": "application/vnd.google-apps.shortcut",
    "owners": [{"displayName": "John Do"}],
    "webViewLink": "https://drive.google.com/file/d/7/view?usp=drivesdk",
    "webContentLink": "https://drive.google.com/uc?id=7&export=download",
    "description": "",
    "modifiedTime": "2023-01-01T00:00:00.0Z",
    "shortcutDetails": {
        "targetId": "1",
        "targetMimeType": "application/vnd.google-apps.document",
    },
}
_text_text = {
    "id": "200",
    "name": "text.txt",
    "mimeType": "text/plain",
    "owners": [{"displayName": "John Do"}],
    "webViewLink": "https://drive.google.com/file/d/200/view?usp=drivesdk",
    "webContentLink": "https://drive.google.com/uc?id=200&export=download",
    "description": "A text",
    "modifiedTime": "2023-01-01T00:00:00.0Z",
    "sha256Checksum": "0000",
}

_application_word = {
    "id": "106",
    "name": "vnd.openxmlformats-officedocument.wordprocessingml.document.docx",
    "mimeType": "application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document",
    "owners": [{"displayName": "John Do"}],
    "webViewLink": "https://docs.google.com/document/d/106/edit?usp=drivesdk&sd=true",
    "webContentLink": "https://drive.google.com/uc?id=106&export=download",
    "description": "A MSWord",
    "modifiedTime": "2023-01-01T00:00:00.0Z",
    "sha256Checksum": "0000",
}
# _application_excel = {
#     "id": "100",
#     "name": "vnd.openxmlformats-officedocument.spreadsheetml.sheet.xlsx",
#     "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#     "owners": [{"displayName": "John Do"}],
#     "webViewLink": "https://docs.google.com/spreadsheets/d/100/edit?"
#     "usp=drivesdk&rtpof=true&sd=true",
#     "webContentLink": "https://drive.google.com/uc?id=100&export=download",
#     "description": "A MSExcel",
#     "modifiedTime": "2023-01-01T00:00:00.0Z",
#     "sha256Checksum": "0000",
# }
# _application_powerpoint = {
#     "id": "103",
#     "name": "vnd.openxmlformats-officedocument.presentationml.presentation.pptx",
#     "mimeType": "application/vnd.openxmlformats-officedocument."
#     "presentationml.presentation",
#     "owners": [{"displayName": "John Do"}],
#     "webViewLink": "https://docs.google.com/presentation/d/103/edit?"
#     "usp=drivesdk&rtpof=true&sd=true",
#     "webContentLink": "https://drive.google.com/uc?id=103&export=download",
#     "description": "A MSPowerpoint",
#     "modifiedTime": "2023-01-01T00:00:00.0Z",
#     "sha256Checksum": "0000",
# }
# _application_ods = {
#     "id": "104",
#     "name": "vnd.oasis.opendocument.spreadsheet.ods",
#     "mimeType": "application/vnd.oasis.opendocument.spreadsheet",
#     "owners": [{"displayName": "John Do"}],
#     "webViewLink": "https://drive.google.com/file/d/104/view?usp=drivesdk",
#     "webContentLink": "https://drive.google.com/uc?id=104&export=download",
#     "description": "A Open Office spreadsheet",
#     "modifiedTime": "2023-01-01T00:00:00.0Z",
#     "sha256Checksum": "0000",
# }
# _application_odt = {
#     "id": "105",
#     "name": "vnd.oasis.opendocument.text.odt",
#     "mimeType": "application/vnd.oasis.opendocument.text",
#     "owners": [{"displayName": "John Do"}],
#     "webViewLink": "https://drive.google.com/file/d/105/view?usp=drivesdk",
#     "webContentLink": "https://drive.google.com/uc?id=105&export=download",
#     "description": "A OpenOffice word",
#     "modifiedTime": "2023-01-01T00:00:00.0Z",
#     "sha256Checksum": "0000",
# }
# _application_odp = {
#     "id": "109",
#     "name": "vnd.oasis.opendocument.presentation.odp",
#     "mimeType": "application/vnd.oasis.opendocument.presentation",
#     "owners": [{"displayName": "John Do"}],
#     "webViewLink": "https://drive.google.com/file/d/109/view?usp=drivesdk",
#     "webContentLink": "https://drive.google.com/uc?id=109&export=download",
#     "description": "A OpenDocument Presentation",
#     "modifiedTime": "2023-01-01T00:00:00.0Z",
#     "sha256Checksum": "0000",
# }
#
# _application_rtf = {
#     "id": "107",
#     "name": "rtf.rtf",
#     "mimeType": "application/rtf",
#     "owners": [{"displayName": "John Do"}],
#     "webViewLink": "https://drive.google.com/file/d/107/view?usp=drivesdk",
#     "webContentLink": "https://drive.google.com/uc?id=107&export=download",
#     "description": "A RTF",
#     "modifiedTime": "2023-01-01T00:00:00.0Z",
#     "sha256Checksum": "0000",
# }
# _application_pdf = {
#     "id": "108",
#     "name": "pdf.pdf",
#     "mimeType": "application/pdf",
#     "owners": [{"displayName": "John Do"}],
#     "webViewLink": "https://drive.google.com/file/d/108/view?usp=drivesdk",
#     "webContentLink": "https://drive.google.com/uc?id=108&export=download",
#     "description": "A PDF",
#     "modifiedTime": "2023-01-01T00:00:00.0Z",
#     "sha256Checksum": "0000",
# }
# _application_json = {
#     "id": "101",
#     "name": "json.json",
#     "mimeType": "application/json",
#     "owners": [{"displayName": "John Do"}],
#     "webViewLink": "https://drive.google.com/file/d/101/view?usp=drivesdk",
#     "webContentLink": "https://drive.google.com/uc?id=101&export=download",
#     "modifiedTime": "2023-01-01T00:00:00.0Z",
#     "sha256Checksum": "0000",
# }
# _application_epub = {
#     "id": "102",
#     "name": "epub+zip.epub",
#     "mimeType": "application/epub+zip",
#     "owners": [{"displayName": "John Do"}],
#     "webViewLink": "https://drive.google.com/file/d/102/view?usp=drivesdk",
#     "webContentLink": "https://drive.google.com/uc?id=102&export=download",
#     "modifiedTime": "2023-01-01T00:00:00.0Z",
#     "sha256Checksum": "0000",
# }

_all_files = {
    _application_vnd_doc["id"]: _application_vnd_doc,
    _application_vnd_slide["id"]: _application_vnd_slide,
    _application_vnd_sheet["id"]: _application_vnd_sheet,
    _application_vnd_form["id"]: _application_vnd_form,
    _application_vnd_jam["id"]: _application_vnd_jam,
    _application_vnd_drawing["id"]: _application_vnd_drawing,
    _application_vnd_shortcut["id"]: _application_vnd_shortcut,
    _application_word["id"]: _application_word,
    # _application_excel["id"]: _application_excel,
    # _application_powerpoint["id"]: _application_powerpoint,
    # _application_ods["id"]: _application_ods,
    # _application_odt["id"]: _application_odt,
    # _application_odp["id"]: _application_odp,
    # _application_rtf["id"]: _application_rtf,
    # _application_pdf["id"]: _application_pdf,
    # _application_json["id"]: _application_json,
    # _application_epub["id"]: _application_epub,
}

gdrive_docs = list(
    filter(
        cast(
            Callable[[Any], bool],
            lambda d: d["mimeType"].startswith("application/vnd.google-apps."),
        ),
        _all_files.values(),
    )
)

not_gdrive_docs = list(
    filter(
        cast(
            Callable[[Any], bool],
            lambda d: not d["mimeType"].startswith("application/vnd.google-apps."),
        ),
        _all_files.values(),
    )
)


def patch_google_workspace(
    mocker: MockerFixture,
    files_result: List[Dict] = [{"nextPageToken": None, "files": gdrive_docs}],
) -> MagicMock:
    """Patch google API with a specific list of files"""
    if not google_workspace_installed:
        return MagicMock()
    import logging

    log_level = logging.DEBUG
    logging.getLogger("langchain.document_loaders.google_drive").setLevel(log_level)
    logging.getLogger("langchain.utilities.google_drive").setLevel(log_level)
    logging.getLogger("langchain.tools.google_drive").setLevel(log_level)

    # Patch Google API authentication
    mock_service_account_credentials = mocker.patch(
        "google.oauth2.service_account.Credentials"
    )
    mock_service_account_credentials.from_service_account_file.return_value = MockCred(
        scopes=SCOPES
    )
    mock_service_account_credentials.from_authorized_user_file.return_value = MockCred(
        scopes=SCOPES
    )
    mock_credentials = mocker.patch("google.oauth2.credentials.Credentials")
    mock_credentials.from_service_account_file.return_value = MockCred(scopes=SCOPES)
    mock_credentials.from_authorized_user_file.return_value = MockCred(scopes=SCOPES)

    # Patch Google API http
    mock_media_io_download = mocker.patch("googleapiclient.http.MediaIoBaseDownload")

    def patch_media_download(*args: List[Any], **kwargs: Any) -> MagicMock:
        fh = cast(io.FileIO, args[0])
        with open(Path(__file__).parent / "examples" / "text.txt", "br") as f:
            buf = f.read()
        fh.write(buf)
        x = MagicMock()
        x.next_chunk.return_value = (0, True)
        return x

    mock_media_io_download.side_effect = patch_media_download
    # Patch Google API build
    mock_googleapiclient_discovery_build = mocker.patch(
        "googleapiclient.discovery.build"
    )
    with open(Path(__file__).parent / "examples" / "gdrive.gdoc") as f:
        gdoc_file = json.load(f)
    with open(Path(__file__).parent / "examples" / "gdrive.gslide") as f:
        gslide_file = json.load(f)
    with open(Path(__file__).parent / "examples" / "gdrive.gsheet") as f:
        gsheet_file = json.load(f)

    def patch_build(*args: List[Any], **kwargs: Any) -> MagicMock:
        if args[0] == "drive":
            drive = MagicMock()
            drive.files.return_value.list.return_value.execute.side_effect = iter(
                files_result
            )

            def patch_get(**kwargs: Any) -> MagicMock:
                x = MagicMock()
                if kwargs["fileId"] in _all_files:
                    x.execute.return_value = _all_files.get(kwargs["fileId"])
                    return x
                else:

                    class FakeResp(BaseModel):
                        status = 400
                        reason: str = ""

                    raise HttpError(
                        FakeResp(),
                        bytes(
                            f"Invalid Value `{kwargs['fileId']}` not found", "utf-8)"
                        ),
                    )

            drive.files.return_value.get.side_effect = patch_get
            return drive
        elif args[0] == "docs":
            docs = MagicMock()
            docs.documents.return_value.get.return_value.execute.return_value = (
                gdoc_file
            )
            return docs
        elif args[0] == "sheets":
            sheet = MagicMock()
            sheet.spreadsheets.return_value.get.return_value.execute.return_value = (
                gsheet_file
            )
            get = sheet.spreadsheets.return_value.values.return_value.get
            get.return_value.execute.return_value = {
                "values": [["a", "b"], [1, 2], [3, 4]]
            }
            return sheet
        elif args[0] == "slides":
            slides = MagicMock()
            slides.presentations.return_value.get.return_value.execute.return_value = (
                gslide_file
            )
            return slides
        else:
            assert False, f"serviceName `{args[0]}` not supported"

    mock_googleapiclient_discovery_build.side_effect = patch_build
    return mock_googleapiclient_discovery_build


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_returns_list_of_google_documents_single(
    google_workspace: MagicMock,
) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        gsheet_mode="single",
        gslide_mode="single",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 3


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_gslide_mode_single(mocker: MockerFixture) -> None:
    patch_google_workspace(
        mocker, [{"nextPageToken": None, "files": [_application_vnd_slide]}]
    )
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        gslide_mode="single",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 1


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_gslide_mode_slide(mocker: MockerFixture) -> None:
    patch_google_workspace(
        mocker, [{"nextPageToken": None, "files": [_application_vnd_slide]}]
    )
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        gslide_mode="slide",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 2


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_gslide_mode_elements(mocker: MockerFixture) -> None:
    patch_google_workspace(
        mocker, [{"nextPageToken": None, "files": [_application_vnd_slide]}]
    )
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        gslide_mode="elements",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 4


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_gsheet_mode_elements(mocker: MockerFixture) -> None:
    patch_google_workspace(
        mocker, [{"nextPageToken": None, "files": [_application_vnd_sheet]}]
    )
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        gsheet_mode="elements",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 2 * 2  # 2 sheet with array of 2 lines


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_gsheet_mode_single(mocker: MockerFixture) -> None:
    patch_google_workspace(
        mocker, [{"nextPageToken": None, "files": [_application_vnd_sheet]}]
    )
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        gsheet_mode="single",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 1


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_num_results_1(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        num_results=1,
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 1


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_mode_snippets(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        mode="snippets",
    )
    assert [doc.page_content for doc in utilities.lazy_get_relevant_documents()] == [
        "It is a doc summary",
        "A slide show",
        "A GSheet",
        "A GForm",
        "A GJam",
        "A GDrawing",
    ]


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_mode_documents(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        mode="documents",
        gsheet_mode="single",
        gslide_mode="single",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert docs[0].page_content.startswith("Body of google docs")
    assert docs[1].page_content.startswith("Title of the presentation")


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_link_field(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        mode="documents",
        gsheet_mode="single",
        gslide_mode="single",
        link_field="webViewLink",
        num_results=1,
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert (
        docs[0].metadata["source"]
    ) == "https://docs.google.com/document/d/1/edit?usp=drivesdk"

    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        mode="documents",
        gsheet_mode="single",
        gslide_mode="single",
        link_field="webContentLink",
        num_results=1,
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert (
        docs[0].metadata["source"]
    ) == "https://drive.google.com/uc?id=1&export=download"


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_document_with_no_link_in_file(mocker: MockerFixture) -> None:
    patch_google_workspace(
        mocker,
        [
            {
                "nextPageToken": None,
                "files": [
                    {
                        "id": "1",
                        "name": "vnd.google-apps.document",
                        "mimeType": "application/vnd.google-apps.document",
                        "owners": [{"displayName": "John Do"}],
                        "description": "It is a doc summary",
                        "modifiedTime": "2023-01-01T00:00:00.0Z",
                    }
                ],
            }
        ],
    )
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        mode="documents",
        gsheet_mode="single",
        gslide_mode="single",
        link_field="webViewLink",
        num_results=1,
    )
    docs = list(utilities.lazy_get_relevant_documents())

    assert (
        docs[0].metadata["source"]
        == "https://docs.google.com/document/d/1/edit?usp=drivesdk"
    )
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        mode="documents",
        gsheet_mode="single",
        gslide_mode="single",
        link_field="webContentLink",  # Literal["webViewLink", "webContentLink"]
        num_results=1,
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert (
        docs[0].metadata["source"]
        == "https://docs.google.com/document/uc?1&export=download"
    )


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_template_query(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-query",
        query="hello",
        gslide_mode="single",
        gsheet_mode="single",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 3


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_template_query_with_mime_type_and_folders(
    google_workspace: MagicMock,
) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        query="toto",
        folder_id=mime_application_folder,
        mime_type="application/vnd.google-apps.document",
        template="gdrive-query-with-mime-type-and-folders",
        gslide_mode="single",
        gsheet_mode="single",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 3


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_custom_template(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        template=PromptTemplate(input_variables=["toto"], template="{toto}"),
        toto="toto",
        gslide_mode="single",
        gsheet_mode="single",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 3


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_default_conv_mapping(mocker: MockerFixture) -> None:
    patch_google_workspace(mocker, [{"nextPageToken": None, "files": [_text_text]}])

    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        template="gdrive-all-in-folders",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0].page_content == "The body of a text file"


try:
    from langchain.document_loaders import TextLoader

    text_loader_installed = True
except ImportError:
    text_loader_installed = False


@unittest.skipIf(not text_loader_installed, "TextLoader not installed")
@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_conv_mapping(mocker: MockerFixture) -> None:
    patch_google_workspace(mocker, [{"nextPageToken": None, "files": [_text_text]}])
    my_mime_types_mapping: TYPE_CONV_MAPPING = {
        "text/plain": TextLoader,
    }
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        conv_mapping=my_mime_types_mapping,
        template="gdrive-all-in-folders",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    print(docs)
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0].page_content == "The body of a text file"


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_with_recursive_shortcut_and_page_token(mocker: MockerFixture) -> None:
    patch_google_workspace(
        mocker,
        [
            {"nextPageToken": "1", "files": [_application_vnd_doc]},
            {"nextPageToken": None, "files": [_application_vnd_folder]},
            {
                "nextPageToken": "2",
                "files": [
                    {
                        **_application_vnd_shortcut,
                        **{
                            "shortcutDetails": {
                                "targetId": _application_vnd_folder["id"],
                                "targetMimeType": "application/vnd.google-apps.folder",
                            }
                        },
                    }
                ],
            },
            {"nextPageToken": None, "files": [_application_vnd_folder2]},
            {"nextPageToken": None, "files": [_application_vnd_slide]},
            {"nextPageToken": None, "files": []},
            {"nextPageToken": None, "files": [_application_vnd_sheet]},
            {"nextPageToken": None, "files": []},
        ],
    )
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        recursive=True,
        template="gdrive-all-in-folders",
        gslide_mode="single",
        gsheet_mode="single",
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 3


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_follow_shortcut_false(mocker: MockerFixture) -> None:
    patch_google_workspace(
        mocker,
        [
            {
                "nextPageToken": None,
                "files": [_application_vnd_shortcut, _application_vnd_doc],
            }
        ],
    )
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        follow_shortcut=False,
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 1


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_follow_shortcut_true(mocker: MockerFixture) -> None:
    # Use the same target document, so return only one document
    patch_google_workspace(
        mocker,
        [
            {
                "nextPageToken": None,
                "files": [_application_vnd_shortcut, _application_vnd_doc],
            }
        ],
    )
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        template="gdrive-all-in-folders",
        folder_id=mime_application_folder,
        follow_shortcut=True,
    )
    docs = list(utilities.lazy_get_relevant_documents())
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 1


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_files_customs_google_api(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        template="gdrive-all-in-folders",
        corpora="user",
        driveId="1",
        fields="id",
        includeItemsFromAllDrives=True,
        includePermissionsForView="published",
        orderBy="name",
        pageSize=10,
        spaces="drive",
        includeLabels=True,
        supportsAllDrives=True,
    )
    list(utilities.lazy_get_relevant_documents())
    utilities._files.list.assert_called_once_with(
        **{
            "corpora": "user",
            "driveId": "1",
            "fields": "nextPageToken, files(id)",
            "includeItemsFromAllDrives": True,
            "includeLabels": True,
            "includePermissionsForView": "published",
            "orderBy": "name",
            "pageSize": 50,
            "spaces": "drive",
            "supportsAllDrives": True,
            "q": f"  '{mime_application_folder}' in parents and trashed=false ",
            "pageToken": None,
        }
    )


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_filter(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        filter=lambda self, file: False,
        template="gdrive-all-in-folders",
    )
    docs = utilities.load_file_from_id(file_id="1")
    assert isinstance(docs, list)
    assert not len(docs)


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_document_from_id(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        template="gdrive-all-in-folders",
    )
    docs = utilities.load_document_from_id(file_id="1")
    assert isinstance(docs, list)


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_document_from_bad_id(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        template="gdrive-all-in-folders",
    )
    docs = utilities.load_document_from_id(file_id="-1")
    assert not docs


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_slides_from_id(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        template="gdrive-all-in-folders",
    )
    docs = utilities.load_slides_from_id(file_id="2")
    assert isinstance(docs, list)


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_slides_from_bad_id(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        template="gdrive-all-in-folders",
    )
    docs = utilities.load_slides_from_id(file_id="-1")
    assert isinstance(docs, list)


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_sheets_from_id(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        template="gdrive-all-in-folders",
    )
    docs = utilities.load_sheets_from_id(file_id="3")
    assert isinstance(docs, list)


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_sheets_from_bad_id(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        template="gdrive-all-in-folders",
    )
    docs = utilities.load_sheets_from_id(file_id="-1")
    assert isinstance(docs, list)


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_file_from_id(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        template="gdrive-all-in-folders",
    )
    docs = utilities.load_file_from_id(file_id="1")
    assert isinstance(docs, list)


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_load_file_from_bad_id(google_workspace: MagicMock) -> None:
    utilities = GoogleDriveUtilities(
        api_file=Path(__file__).parent / "examples" / "gdrive_credentials.json",
        folder_id=mime_application_folder,
        template="gdrive-all-in-folders",
    )
    docs = utilities.load_file_from_id(file_id="-1")
    assert isinstance(docs, list)


def test_snippet_from_page_content() -> None:
    assert GoogleDriveUtilities._snippet_from_page_content("ABCDEF", 6) == "ABCDEF"
    assert GoogleDriveUtilities._snippet_from_page_content("ABCDEFG", 6) == "ABC..."
    assert GoogleDriveUtilities._snippet_from_page_content("ABCDEFGHI", 6) == "ABC..."
    assert (
        GoogleDriveUtilities._snippet_from_page_content("ABCDEFGHIJ", 6) == "ABC...HIJ"
    )


# %% ----------------
@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_default(google_workspace: MagicMock) -> None:
    wrapper = GoogleDriveAPIWrapper(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
    )
    assert wrapper.gslide_mode == "single"
    assert wrapper.gsheet_mode == "single"


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_query_snippets(google_workspace: MagicMock) -> None:
    wrapper = GoogleDriveAPIWrapper(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        mode="snippets",
    )
    result = wrapper.run("machine learning")
    assert result.startswith(
        "Name: vnd.google-apps.document\n"
        "Source: https://docs.google.com/document/d/1/edit?usp=drivesdk\n"
        "Summary: It is a doc summary\n\n"
    )


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_query_snippets_markdown(google_workspace: MagicMock) -> None:
    wrapper = GoogleDriveAPIWrapper(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        mode="snippets-markdown",
    )
    result = wrapper.run("machine learning")
    assert result.startswith(
        "[vnd.google-apps.document](https://docs.google.com/document/d/1/edit?"
        "usp=drivesdk)<br/>\n"
        "It is a doc summary\n\n"
    )


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_query_documents(google_workspace: MagicMock) -> None:
    wrapper = GoogleDriveAPIWrapper(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        mode="documents",
    )
    result = wrapper.run("machine learning")
    assert result.startswith(
        "Name: vnd.google-apps.document\n"
        "Source: https://docs.google.com/document/d/1/edit?usp=drivesdk\n"
        "Summary: Body of google docs."
    )


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_query_documents_markdown(google_workspace: MagicMock) -> None:
    wrapper = GoogleDriveAPIWrapper(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        mode="documents-markdown",
    )
    result = wrapper.run("machine learning")
    assert result.startswith(
        "[vnd.google-apps.document](https://docs.google.com/document/d/1/edit?"
        "usp=drivesdk)<br/>"
        "Body of google docs."
    )


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_query_no_documents(mocker: MockerFixture) -> None:
    patch_google_workspace(mocker, [{"nextPageToken": None, "files": []}])
    wrapper = GoogleDriveAPIWrapper(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        mode="documents-markdown",
    )
    result = wrapper.run("machine learning")
    assert result == "No document found"


@unittest.skipIf(not google_workspace_installed, "Google api not installed")
def test_no_query(google_workspace: MagicMock) -> None:
    wrapper = GoogleDriveAPIWrapper(
        api_file=Path(__file__).parent.parent
        / "utilities"
        / "examples"
        / "gdrive_credentials.json",
        mode="documents-markdown",
    )
    wrapper.run("")
    assert "''" not in wrapper.files.list.call_args[1]["q"]


# Private test with a directory with a sample of all file format
# @unittest.skipIf(not google_workspace_installed, "Google api not installed")
# def test_load_mime() -> None:
#     utilities = GoogleDriveUtilities(
#         folder_id='18A21b37hPISOQtStQ_irQLYS3hlVEsBH',
#         template="gdrive-all-in-folders",
#         # template="gdrive-mime-type-in-folders", mime_type="image/png",
#         recursive=True,
#     )
#     docs = list(utilities.lazy_get_relevant_documents())
#     print(docs)
