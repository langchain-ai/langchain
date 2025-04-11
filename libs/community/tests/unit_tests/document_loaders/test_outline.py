from typing import Dict

import pytest
import requests
import requests_mock

from langchain_community.document_loaders import OutlineLoader


@pytest.fixture
def outline_loader() -> OutlineLoader:
    return OutlineLoader(
        outline_base_url="http://outline.test", outline_api_key="test-api-key"
    )


@pytest.fixture
def mock_response_single_page() -> Dict:
    return {
        "data": [
            {
                "id": "1",
                "text": "Test document 1",
                "title": "Test 1",
                "createdAt": "2024-03-26T20:00:01.781Z",
                "updatedAt": "2024-03-26T20:00:01.781Z",
                "url": "/doc/test-RTYIxmoduo",
            }
        ],
        "pagination": {
            "nextPath": "/api/documents.list?limit=25&offset=25",
            "total": 1,
        },
    }


@pytest.fixture
def mock_response_multiple_pages() -> Dict:
    return {
        "data": [
            {
                "id": "1",
                "text": "Test document 1",
                "title": "Test 1",
                "createdAt": "2024-03-26T20:00:01.781Z",
                "updatedAt": "2024-03-26T20:00:01.781Z",
                "url": "/doc/test-RTYIxmoduo",
            }
        ],
        "pagination": {
            "nextPath": "/api/documents.list?limit=25&offset=25",
            "total": 30,
        },
    }


def test_fetch_single_page(
    outline_loader: OutlineLoader, mock_response_single_page: Dict
) -> None:
    with requests_mock.Mocker() as m:
        m.post("http://outline.test/api/documents.list", json=mock_response_single_page)

        documents = outline_loader.load()

        assert len(documents) == 1
        assert documents[0].page_content == "Test document 1"


def test_fetch_multiple_pages(
    outline_loader: OutlineLoader, mock_response_multiple_pages: Dict
) -> None:
    with requests_mock.Mocker() as m:
        # Second page (last page)
        second_page = {
            "data": [
                {
                    "id": "2",
                    "text": "Test document 2",
                    "title": "Test 2",
                    "createdAt": "2024-03-26T20:00:01.781Z",
                    "updatedAt": "2024-03-26T20:00:01.781Z",
                    "url": "/doc/test-RTYIxmodua",
                }
            ],
            "pagination": {
                "nextPath": "http://outline.test/api/documents.list?limit=25&offset=50",
                "total": 30,
            },
        }
        # First page
        m.post(
            "http://outline.test/api/documents.list",
            [{"json": mock_response_multiple_pages}, {"json": second_page}],
        )

        documents = outline_loader.load()

        assert len(documents) == 2
        assert documents[0].page_content == "Test document 1"
        assert documents[1].page_content == "Test document 2"


def test_api_error(outline_loader: OutlineLoader) -> None:
    with requests_mock.Mocker() as m:
        m.post("http://outline.test/api/documents.list", status_code=401)

        with pytest.raises(requests.exceptions.HTTPError):
            outline_loader.load()
