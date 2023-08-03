import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest as pytest
from pytest_mock import MockerFixture

from langchain.tools.google_drive.tool import GoogleDriveSearchTool
from langchain.utilities import GoogleDriveAPIWrapper
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
def test_run(google_workspace: MagicMock) -> None:
    tool = GoogleDriveSearchTool(
        api_wrapper=GoogleDriveAPIWrapper(
            api_file=(
                Path(__file__).parent.parent
                / "utilities"
                / "examples"
                / "gdrive_credentials.json"
            )
        )
    )
    result = tool._run("machine learning")
    assert result.startswith(
        "[vnd.google-apps.document](https://docs.google.com/document/d/1/edit?usp=drivesdk)<br/>\n"
        "It is a doc summary\n\n"
    )
