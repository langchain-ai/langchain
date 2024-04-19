"""Integration test for Outline API Wrapper."""

from typing import List

import pytest
import responses
from langchain_core.documents import Document

from langchain_community.utilities import OutlineAPIWrapper

OUTLINE_INSTANCE_TEST_URL = "https://app.getoutline.com"
OUTLINE_SUCCESS_RESPONSE = {
    "data": [
        {
            "ranking": 0.3911583,
            "context": "Testing Context",
            "document": {
                "id": "abb2bf15-a597-4255-8b19-b742e3d037bf",
                "url": "/doc/quick-start-jGuGGGOTuL",
                "title": "Test Title",
                "text": "Testing Content",
                "createdBy": {"name": "John Doe"},
                "revision": 3,
                "collectionId": "93f182a4-a591-4d47-83f0-752e7bb2065c",
                "parentDocumentId": None,
            },
        },
    ],
    "status": 200,
    "ok": True,
}

OUTLINE_EMPTY_RESPONSE = {
    "data": [],
    "status": 200,
    "ok": True,
}

OUTLINE_ERROR_RESPONSE = {
    "ok": False,
    "error": "authentication_required",
    "status": 401,
    "message": "Authentication error",
}


@pytest.fixture
def api_client() -> OutlineAPIWrapper:
    return OutlineAPIWrapper(
        outline_api_key="api_key", outline_instance_url=OUTLINE_INSTANCE_TEST_URL
    )


def assert_docs(docs: List[Document], all_meta: bool = False) -> None:
    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        main_meta = {"title", "source"}
        assert set(doc.metadata).issuperset(main_meta)
        if all_meta:
            assert len(set(doc.metadata)) > len(main_meta)
        else:
            assert len(set(doc.metadata)) == len(main_meta)


@responses.activate
def test_run_success(api_client: OutlineAPIWrapper) -> None:
    responses.add(
        responses.POST,
        api_client.outline_instance_url + api_client.outline_search_endpoint,  # type: ignore[operator]
        json=OUTLINE_SUCCESS_RESPONSE,
        status=200,
    )

    docs = api_client.run("Testing")
    assert_docs(docs, all_meta=False)


@responses.activate
def test_run_success_all_meta(api_client: OutlineAPIWrapper) -> None:
    api_client.load_all_available_meta = True
    responses.add(
        responses.POST,
        api_client.outline_instance_url + api_client.outline_search_endpoint,  # type: ignore[operator]
        json=OUTLINE_SUCCESS_RESPONSE,
        status=200,
    )

    docs = api_client.run("Testing")
    assert_docs(docs, all_meta=True)


@responses.activate
def test_run_no_result(api_client: OutlineAPIWrapper) -> None:
    responses.add(
        responses.POST,
        api_client.outline_instance_url + api_client.outline_search_endpoint,  # type: ignore[operator]
        json=OUTLINE_EMPTY_RESPONSE,
        status=200,
    )

    docs = api_client.run("No Result Test")
    assert not docs


@responses.activate
def test_run_error(api_client: OutlineAPIWrapper) -> None:
    responses.add(
        responses.POST,
        api_client.outline_instance_url + api_client.outline_search_endpoint,  # type: ignore[operator]
        json=OUTLINE_ERROR_RESPONSE,
        status=401,
    )
    try:
        api_client.run("Testing")
    except Exception as e:
        assert "Outline API returned an error:" in str(e)
