from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from pytest_mock import MockerFixture

from langchain_box.utilities import BoxAPIWrapper


@pytest.fixture()
def mock_worker(mocker: MockerFixture) -> None:
    mocker.patch("langchain_box.utilities.BoxAuth.authorize", return_value=Mock())
    mocker.patch("langchain_box.utilities.BoxAuth.get_client", return_value=Mock())
    mocker.patch(
        "langchain_box.utilities.BoxAPIWrapper.get_text_representation",
        return_value=("filename", "content", "url"),
    )


def test_get_documents_by_file_ids(mock_worker, mocker: MockerFixture) -> None:  # type: ignore[no-untyped-def]
    mocker.patch(
        "langchain_box.utilities.BoxAPIWrapper.get_document_by_file_id",
        return_value=(
            Document(
                page_content="content", metadata={"source": "url", "title": "filename"}
            )
        ),
    )

    box = BoxAPIWrapper(box_developer_token="box_developer_token")  # type: ignore[call-arg]

    documents = box.get_document_by_file_id("box_file_id")
    assert documents == Document(
        page_content="content", metadata={"source": "url", "title": "filename"}
    )


def test_get_documents_by_folder_id(mock_worker, mocker: MockerFixture) -> None:  # type: ignore[no-untyped-def]
    mocker.patch(
        "langchain_box.utilities.BoxAPIWrapper.get_folder_items",
        return_value=([{"id": "file_id", "type": "file"}]),
    )

    box = BoxAPIWrapper(box_developer_token="box_developer_token")  # type: ignore[call-arg]

    folder_contents = box.get_folder_items("box_folder_id")
    assert folder_contents == [{"id": "file_id", "type": "file"}]
