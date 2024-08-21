import pytest

from langchain_box.document_loaders import BoxLoader
from langchain_box.utilities import BoxAuth, BoxAuthType


# Test auth types
def test_direct_token_initialization() -> None:
    loader = BoxLoader(  # type: ignore[call-arg]
        box_developer_token="box_developer_token",
        box_file_ids=["box_file_ids"],
    )

    assert loader.box_developer_token == "box_developer_token"
    assert loader.box_file_ids == ["box_file_ids"]


def test_failed_direct_token_initialization() -> None:
    with pytest.raises(ValueError):
        loader = BoxLoader(box_file_ids=["box_file_ids"])  # type: ignore[call-arg] # noqa: F841


def test_auth_initialization() -> None:
    auth = BoxAuth(
        auth_type=BoxAuthType.TOKEN, box_developer_token="box_developer_token"
    )

    loader = BoxLoader(  # type: ignore[call-arg]
        box_auth=auth,
        box_file_ids=["box_file_ids"],
    )

    assert loader.box_file_ids == ["box_file_ids"]


# test loaders
def test_failed_file_initialization() -> None:
    with pytest.raises(ValueError):
        loader = BoxLoader(box_developer_token="box_developer_token")  # type: ignore[call-arg] # noqa: F841


def test_folder_initialization() -> None:
    loader = BoxLoader(  # type: ignore[call-arg]
        box_developer_token="box_developer_token",
        box_folder_id="box_folder_id",
    )

    assert loader.box_developer_token == "box_developer_token"
    assert loader.box_folder_id == "box_folder_id"


def test_failed_initialization_files_and_folders() -> None:
    with pytest.raises(ValueError):
        loader = BoxLoader(  # type: ignore[call-arg] # noqa: F841
            box_developer_token="box_developer_token",
            box_folder_id="box_folder_id",
            box_file_ids=["box_file_ids"],
        )
