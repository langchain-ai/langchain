import os
from typing import Any
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from langchain_core.documents import Document
from pytest_mock import MockerFixture

from langchain_community.document_loaders.onenote import OneNoteLoader


def test_initialization() -> None:
    os.environ["MS_GRAPH_CLIENT_ID"] = "CLIENT_ID"
    os.environ["MS_GRAPH_CLIENT_SECRET"] = "CLIENT_SECRET"

    loader = OneNoteLoader(
        notebook_name="test_notebook",
        section_name="test_section",
        page_title="test_title",
        access_token="access_token",
    )
    assert loader.notebook_name == "test_notebook"
    assert loader.section_name == "test_section"
    assert loader.page_title == "test_title"
    assert loader.access_token == "access_token"
    assert loader._headers == {
        "Authorization": "Bearer access_token",
    }


@pytest.mark.requires("bs4")
def test_load(mocker: MockerFixture) -> None:
    os.environ["MS_GRAPH_CLIENT_ID"] = "CLIENT_ID"
    os.environ["MS_GRAPH_CLIENT_SECRET"] = "CLIENT_SECRET"

    mocker.patch(
        "requests.get",
        return_value=mocker.MagicMock(json=lambda: {"value": []}, links=None),
    )
    loader = OneNoteLoader(
        notebook_name="test_notebook",
        section_name="test_section",
        page_title="test_title",
        access_token="access_token",
    )
    documents = loader.load()
    assert documents == []

    mocker.patch(
        "langchain_community.document_loaders.onenote.OneNoteLoader._get_page_content",
        return_value=(
            "<html><head><title>Test Title</title></head>"
            "<body><p>Test Content</p></body></html>"
        ),
    )
    loader = OneNoteLoader(object_ids=["test_id"], access_token="access_token")
    documents = loader.load()
    assert documents == [
        Document(
            page_content="Test Title\nTest Content", metadata={"title": "Test Title"}
        )
    ]


class FakeConfidentialClientApplication(Mock):
    def get_authorization_request_url(self, *args: Any, **kwargs: Any) -> str:
        return "fake_authorization_url"


@pytest.mark.requires("msal")
def test_msal_import(monkeypatch: MonkeyPatch, mocker: MockerFixture) -> None:
    os.environ["MS_GRAPH_CLIENT_ID"] = "CLIENT_ID"
    os.environ["MS_GRAPH_CLIENT_SECRET"] = "CLIENT_SECRET"

    monkeypatch.setattr("builtins.input", lambda _: "invalid_url")
    mocker.patch(
        "msal.ConfidentialClientApplication",
        return_value=FakeConfidentialClientApplication(),
    )
    loader = OneNoteLoader(
        notebook_name="test_notebook",
        section_name="test_section",
        page_title="test_title",
    )
    with pytest.raises(IndexError):
        loader._auth()


def test_url() -> None:
    os.environ["MS_GRAPH_CLIENT_ID"] = "CLIENT_ID"
    os.environ["MS_GRAPH_CLIENT_SECRET"] = "CLIENT_SECRET"

    loader = OneNoteLoader(
        notebook_name="test_notebook",
        section_name="test_section",
        page_title="test_title",
        access_token="access_token",
        onenote_api_base_url="https://graph.microsoft.com/v1.0/me/onenote",
    )
    assert loader._url == (
        "https://graph.microsoft.com/v1.0/me/onenote/pages?$select=id"
        "&$expand=parentNotebook,parentSection"
        "&$filter=parentNotebook/displayName%20eq%20'test_notebook'"
        "%20and%20parentSection/displayName%20eq%20'test_section'"
        "%20and%20title%20eq%20'test_title'"
    )

    loader = OneNoteLoader(
        notebook_name="test_notebook",
        section_name="test_section",
        access_token="access_token",
        onenote_api_base_url="https://graph.microsoft.com/v1.0/me/onenote",
    )
    assert loader._url == (
        "https://graph.microsoft.com/v1.0/me/onenote/pages?$select=id"
        "&$expand=parentNotebook,parentSection"
        "&$filter=parentNotebook/displayName%20eq%20'test_notebook'"
        "%20and%20parentSection/displayName%20eq%20'test_section'"
    )

    loader = OneNoteLoader(
        notebook_name="test_notebook",
        access_token="access_token",
        onenote_api_base_url="https://graph.microsoft.com/v1.0/me/onenote",
    )
    assert loader._url == (
        "https://graph.microsoft.com/v1.0/me/onenote/pages?$select=id"
        "&$expand=parentNotebook"
        "&$filter=parentNotebook/displayName%20eq%20'test_notebook'"
    )

    loader = OneNoteLoader(
        section_name="test_section",
        access_token="access_token",
        onenote_api_base_url="https://graph.microsoft.com/v1.0/me/onenote",
    )
    assert loader._url == (
        "https://graph.microsoft.com/v1.0/me/onenote/pages?$select=id"
        "&$expand=parentSection"
        "&$filter=parentSection/displayName%20eq%20'test_section'"
    )

    loader = OneNoteLoader(
        section_name="test_section",
        page_title="test_title",
        access_token="access_token",
        onenote_api_base_url="https://graph.microsoft.com/v1.0/me/onenote",
    )
    assert loader._url == (
        "https://graph.microsoft.com/v1.0/me/onenote/pages?$select=id"
        "&$expand=parentSection"
        "&$filter=parentSection/displayName%20eq%20'test_section'"
        "%20and%20title%20eq%20'test_title'"
    )

    loader = OneNoteLoader(
        page_title="test_title",
        access_token="access_token",
        onenote_api_base_url="https://graph.microsoft.com/v1.0/me/onenote",
    )
    assert loader._url == (
        "https://graph.microsoft.com/v1.0/me/onenote/pages?$select=id"
        "&$filter=title%20eq%20'test_title'"
    )
