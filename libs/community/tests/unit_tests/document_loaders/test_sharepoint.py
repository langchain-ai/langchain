import json
import os
import shutil
import uuid
from pathlib import Path

from pytest_mock import MockerFixture

from langchain_community.document_loaders.base_o365 import _O365TokenStorage
from langchain_community.document_loaders.sharepoint import SharePointLoader

O365_CLIENT_ID = "CLIENT_ID"
O365_CLIENT_SECRET = "CLIENT_SECRET"


def test_initialization() -> None:
    os.environ["O365_CLIENT_ID"] = O365_CLIENT_ID
    os.environ["O365_CLIENT_SECRET"] = O365_CLIENT_SECRET

    loader = SharePointLoader(
        document_library_id="fake_library_id",
    )

    assert loader.document_library_id == "fake_library_id"
    assert loader.settings.client_id == O365_CLIENT_ID
    assert loader.settings.client_secret.get_secret_value() == O365_CLIENT_SECRET


def test_recursive() -> None:
    os.environ["O365_CLIENT_ID"] = O365_CLIENT_ID
    os.environ["O365_CLIENT_SECRET"] = O365_CLIENT_SECRET

    loader = SharePointLoader(document_library_id="fake_library_id", recursive=True)

    assert loader.document_library_id == "fake_library_id"
    assert loader.recursive is True


def test_folder_path() -> None:
    os.environ["O365_CLIENT_ID"] = O365_CLIENT_ID
    os.environ["O365_CLIENT_SECRET"] = O365_CLIENT_SECRET

    loader = SharePointLoader(
        document_library_id="fake_library_id", recursive=True, folder_path="/test"
    )

    assert loader.document_library_id == "fake_library_id"
    assert loader.recursive is True
    assert loader.folder_path == "/test"


def test_object_ids() -> None:
    os.environ["O365_CLIENT_ID"] = O365_CLIENT_ID
    os.environ["O365_CLIENT_SECRET"] = O365_CLIENT_SECRET

    loader = SharePointLoader(
        document_library_id="fake_library_id", recursive=True, object_ids=["1"]
    )

    assert loader.document_library_id == "fake_library_id"
    assert loader.recursive is True
    assert loader.object_ids == ["1"]


def test_auth_with_default_token(mocker: MockerFixture) -> None:
    os.environ["O365_CLIENT_ID"] = O365_CLIENT_ID
    os.environ["O365_CLIENT_SECRET"] = O365_CLIENT_SECRET

    mock_get_current_user = mocker.patch(
        "O365.account.Account.get_current_user",
        return_value=None,
    )

    loader = SharePointLoader(
        document_library_id="fake_library_id", auth_with_token=True
    )

    account = loader._auth()

    mock_get_current_user.assert_called_once()
    assert loader.settings.client_id == O365_CLIENT_ID
    assert loader.settings.client_secret.get_secret_value() == O365_CLIENT_SECRET
    assert account is not None
    assert (
        account.con.token_backend.token_path
        == Path.home() / ".credentials" / "o365_token.txt"
    )


def test_auth_with_token(mocker: MockerFixture) -> None:
    mock_get_current_user = mocker.patch(
        "O365.account.Account.get_current_user",
        return_value=None,
    )
    try:
        # create temporary token file
        temporary_dir = os.path.join(
            "/tmp/.credentials/O365Loader",
            # We use a unique identifier so that we don't risk overwriting
            # existing files
            str(uuid.uuid4()).replace("-", "_"),
        )
        Path(temporary_dir).mkdir(parents=True, exist_ok=True)
        token_path = os.path.join(temporary_dir, "o365_token.txt")
        with open(token_path, "w") as f:
            json.dump(
                {
                    "token_type": "Bearer",
                    "scope": [
                        "profile",
                        "openid",
                        "email",
                        "https://graph.microsoft.com/Sites.Read.All",
                        "https://graph.microsoft.com/User.Read",
                    ],
                    "expires_in": 0,
                    "ext_expires_in": 0,
                    "access_token": "fake_access_token",
                    "refresh_token": "fake_refresh_token",
                    "expires_at": 0,
                },
                f,
            )

        loader = SharePointLoader(
            document_library_id="fake_library_id",
            auth_with_token=True,
            token_storage=_O365TokenStorage(token_path=token_path),
            recursive=True,
        )

        account = loader._auth()

        mock_get_current_user.assert_called_once()
        assert account is not None
        assert account.con.token_backend.token_path.absolute().as_posix() == token_path

    finally:
        shutil.rmtree(temporary_dir)
