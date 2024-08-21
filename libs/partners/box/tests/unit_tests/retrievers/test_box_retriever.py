import pytest
from langchain_core.documents import Document
from pytest_mock import MockerFixture

from langchain_box.retrievers import BoxRetriever
from langchain_box.utilities import BoxAuth, BoxAuthType


# Test auth types
def test_direct_token_initialization() -> None:
    retriever = BoxRetriever(  # type: ignore[call-arg]
        box_developer_token="box_developer_token",
        box_file_ids=["box_file_ids"],
    )

    assert retriever.box_developer_token == "box_developer_token"
    assert retriever.box_file_ids == ["box_file_ids"]


def test_failed_direct_token_initialization() -> None:
    with pytest.raises(ValueError):
        retriever = BoxRetriever(box_file_ids=["box_file_ids"])  # type: ignore[call-arg] # noqa: F841


def test_auth_initialization() -> None:
    auth = BoxAuth(
        auth_type=BoxAuthType.TOKEN, box_developer_token="box_developer_token"
    )

    retriever = BoxRetriever(  # type: ignore[call-arg]
        box_auth=auth,
        box_file_ids=["box_file_ids"],
    )

    assert retriever.box_file_ids == ["box_file_ids"]


# test search retrieval
def test_search(mocker: MockerFixture) -> None:
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

    retriever = BoxRetriever(  # type: ignore[call-arg]
        box_developer_token="box_developer_token"
    )

    documents = retriever.invoke("query")
    assert documents == [
        Document(
            page_content="Test file mode\ndocument contents",
            metadata={"title": "Testing Files"},
        )
    ]


# test ai retrieval
def test_ai(mocker: MockerFixture) -> None:
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

    retriever = BoxRetriever(  # type: ignore[call-arg]
        box_developer_token="box_developer_token", box_file_ids=["box_file_ids"]
    )

    documents = retriever.invoke("query")
    assert documents == [
        Document(
            page_content="Test file mode\ndocument contents",
            metadata={"title": "Testing Files"},
        )
    ]
