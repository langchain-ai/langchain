from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from pydantic.v1.error_wrappers import ValidationError
from pytest_mock import MockerFixture

from langchain_box.utilities import BoxAuth, BoxAuthType, _BoxAPIWrapper


@pytest.fixture()
def mock_worker(mocker: MockerFixture) -> None:
    mocker.patch("langchain_box.utilities.BoxAuth._authorize", return_value=Mock())
    mocker.patch("langchain_box.utilities.BoxAuth.get_client", return_value=Mock())
    mocker.patch(
        "langchain_box.utilities._BoxAPIWrapper._get_text_representation",
        return_value=("filename", "content", "url"),
    )


# Test auth types
def test_token_initialization() -> None:
    auth = BoxAuth(
        auth_type=BoxAuthType.TOKEN, box_developer_token="box_developer_token"
    )

    assert auth.auth_type == "token"
    assert auth.box_developer_token == "box_developer_token"


def test_failed_token_initialization() -> None:
    with pytest.raises(ValidationError):
        auth = BoxAuth(auth_type=BoxAuthType.TOKEN)  # noqa: F841


def test_jwt_eid_initialization() -> None:
    auth = BoxAuth(auth_type=BoxAuthType.JWT, box_jwt_path="box_jwt_path")

    assert auth.auth_type == "jwt"
    assert auth.box_jwt_path == "box_jwt_path"


def test_jwt_user_initialization() -> None:
    auth = BoxAuth(
        auth_type=BoxAuthType.JWT,
        box_jwt_path="box_jwt_path",
        box_user_id="box_user_id",
    )

    assert auth.auth_type == "jwt"
    assert auth.box_jwt_path == "box_jwt_path"
    assert auth.box_user_id == "box_user_id"


def test_failed_jwt_initialization() -> None:
    with pytest.raises(ValidationError):
        auth = BoxAuth(auth_type=BoxAuthType.JWT, box_user_id="box_user_id")  # noqa: F841


def test_ccg_eid_initialization() -> None:
    auth = BoxAuth(
        auth_type=BoxAuthType.CCG,
        box_client_id="box_client_id",
        box_client_secret="box_client_secret",
        box_enterprise_id="box_enterprise_id",
    )

    assert auth.auth_type == "ccg"
    assert auth.box_client_id == "box_client_id"
    assert auth.box_client_secret == "box_client_secret"
    assert auth.box_enterprise_id == "box_enterprise_id"


def test_ccg_user_initialization() -> None:
    auth = BoxAuth(
        auth_type=BoxAuthType.CCG,
        box_client_id="box_client_id",
        box_client_secret="box_client_secret",
        box_enterprise_id="box_enterprise_id",
        box_user_id="box_user_id",
    )

    assert auth.auth_type == "ccg"
    assert auth.box_client_id == "box_client_id"
    assert auth.box_client_secret == "box_client_secret"
    assert auth.box_enterprise_id == "box_enterprise_id"
    assert auth.box_user_id == "box_user_id"


def test_failed_ccg_initialization() -> None:
    with pytest.raises(ValidationError):
        auth = BoxAuth(auth_type=BoxAuthType.CCG)  # noqa: F841


def test_direct_token_initialization() -> None:
    box = _BoxAPIWrapper(  #  type: ignore[call-arg]
        box_developer_token="box_developer_token"
    )

    assert box.box_developer_token == "box_developer_token"


def test_auth_initialization() -> None:
    auth = BoxAuth(
        auth_type=BoxAuthType.TOKEN, box_developer_token="box_developer_token"
    )

    box = _BoxAPIWrapper(box_auth=auth)  #  type: ignore[call-arg] # noqa: F841

    assert auth.box_developer_token == "box_developer_token"


def test_failed_initialization_no_auth() -> None:
    with pytest.raises(ValidationError):
        box = _BoxAPIWrapper()  # type: ignore[call-arg] # noqa: F841


def test_get_documents_by_file_ids(mock_worker, mocker: MockerFixture) -> None:  # type: ignore[no-untyped-def]
    mocker.patch(
        "langchain_box.utilities._BoxAPIWrapper.get_document_by_file_id",
        return_value=(
            Document(
                page_content="content", metadata={"source": "url", "title": "filename"}
            )
        ),
    )

    box = _BoxAPIWrapper(box_developer_token="box_developer_token")  # type: ignore[call-arg]

    documents = box.get_document_by_file_id("box_file_id")
    assert documents == Document(
        page_content="content", metadata={"source": "url", "title": "filename"}
    )


def test_get_documents_by_folder_id(mock_worker, mocker: MockerFixture) -> None:  # type: ignore[no-untyped-def]
    mocker.patch(
        "langchain_box.utilities._BoxAPIWrapper.get_folder_items",
        return_value=([{"id": "file_id", "type": "file"}]),
    )

    box = _BoxAPIWrapper(box_developer_token="box_developer_token")  # type: ignore[call-arg]

    folder_contents = box.get_folder_items("box_folder_id")
    assert folder_contents == [{"id": "file_id", "type": "file"}]


def test_box_search(mock_worker, mocker: MockerFixture) -> None:  # type: ignore[no-untyped-def]
    mocker.patch(
        "langchain_box.utilities._BoxAPIWrapper.search_box",
        return_value=(
            [
                Document(
                    page_content="Test file mode\ndocument contents",
                    metadata={"title": "Testing Files"},
                )
            ]
        ),
    )

    box = _BoxAPIWrapper(box_developer_token="box_developer_token")  # type: ignore[call-arg]

    documents = box.search_box("query")
    assert documents == [
        Document(
            page_content="Test file mode\ndocument contents",
            metadata={"title": "Testing Files"},
        )
    ]


def test_ask_box_ai_single_file(mock_worker, mocker: MockerFixture) -> None:  # type: ignore[no-untyped-def]
    mocker.patch(
        "langchain_box.utilities._BoxAPIWrapper.ask_box_ai",
        return_value=(
            [
                Document(
                    page_content="Test file mode\ndocument contents",
                    metadata={"title": "Testing Files"},
                )
            ]
        ),
    )

    box = _BoxAPIWrapper(  # type: ignore[call-arg]
        box_developer_token="box_developer_token", box_file_ids=["box_file_ids"]
    )

    documents = box.ask_box_ai("query")  #  type: ignore[call-arg]
    assert documents == [
        Document(
            page_content="Test file mode\ndocument contents",
            metadata={"title": "Testing Files"},
        )
    ]


def test_ask_box_ai_multiple_files(mock_worker, mocker: MockerFixture) -> None:  # type: ignore[no-untyped-def]
    mocker.patch(
        "langchain_box.utilities._BoxAPIWrapper.ask_box_ai",
        return_value=(
            [
                Document(
                    page_content="Test file 1 mode\ndocument contents",
                    metadata={"title": "Test File 1"},
                ),
                Document(
                    page_content="Test file 2 mode\ndocument contents",
                    metadata={"title": "Test File 2"},
                ),
            ]
        ),
    )

    box = _BoxAPIWrapper(  # type: ignore[call-arg]
        box_developer_token="box_developer_token",
        box_file_ids=["box_file_id 1", "box_file_id 2"],
    )

    documents = box.ask_box_ai("query")  #  type: ignore[call-arg]
    assert documents == [
        Document(
            page_content="Test file 1 mode\ndocument contents",
            metadata={"title": "Test File 1"},
        ),
        Document(
            page_content="Test file 2 mode\ndocument contents",
            metadata={"title": "Test File 2"},
        ),
    ]
