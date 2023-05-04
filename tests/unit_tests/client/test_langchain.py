"""Test the LangChain+ client."""
from datetime import datetime
from io import BytesIO
from unittest import mock

import pytest

from langchain.client.langchain import LangChainClient, _get_link_stem, _is_localhost
from langchain.client.models import Example

_CREATED_AT = datetime(2015, 1, 1, 0, 0, 0)


@pytest.mark.parametrize(
    "api_url, expected_url",
    [
        ("http://localhost:8000", "http://localhost"),
        ("http://www.example.com", "http://www.example.com"),
        (
            "https://hosted-1234-23qwerty.f.234.foobar.gateway.dev",
            "https://hosted-1234-23qwerty.f.234.foobar.gateway.dev",
        ),
        ("https://www.langchain.com/path/to/nowhere", "https://www.langchain.com"),
    ],
)
def test_link_split(api_url: str, expected_url: str) -> None:
    """Test the link splitting handles both localhost and deployed urls."""
    assert _get_link_stem(api_url) == expected_url


def test_is_localhost() -> None:
    assert _is_localhost("http://localhost:8000")
    assert _is_localhost("http://127.0.0.1:8000")
    assert _is_localhost("http://0.0.0.0:8000")
    assert not _is_localhost("http://example.com:8000")


def test_validate_api_key_if_hosted() -> None:
    with pytest.raises(ValueError, match="API key must be provided"):
        LangChainClient(api_url="http://www.example.com")

    client = LangChainClient(api_url="http://localhost:8000")
    assert client.api_url == "http://localhost:8000"
    assert client.api_key is None


def test_headers() -> None:
    client = LangChainClient(api_url="http://localhost:8000", api_key="123")
    assert client._headers == {"authorization": "Bearer 123"}

    client_no_key = LangChainClient(api_url="http://localhost:8000")
    assert client_no_key._headers == {}


@mock.patch("langchain.client.langchain.requests.post")
def test_create_session(mock_post: mock.Mock) -> None:
    mock_response = mock.Mock()
    mock_response.json.return_value = {"id": 1}
    mock_post.return_value = mock_response

    client = LangChainClient(api_url="http://localhost:8000", api_key="123")
    session = client.create_session("test_session")

    assert session.id == 1
    assert session.name == "test_session"
    mock_post.assert_called_once_with(
        "http://localhost:8000/sessions",
        data=mock.ANY,
        headers={"authorization": "Bearer 123"},
    )


@mock.patch("langchain.client.langchain.requests.post")
def test_upload_csv(mock_post: mock.Mock) -> None:
    mock_response = mock.Mock()
    example_1 = Example(
        id="1",
        created_at=_CREATED_AT,
        inputs={"input": "1"},
        outputs={"output": "2"},
        dataset_id="1",
    )
    example_2 = Example(
        id="2",
        created_at=_CREATED_AT,
        inputs={"input": "3"},
        outputs={"output": "4"},
        dataset_id="1",
    )

    mock_response.json.return_value = {
        "id": "1",
        "name": "test.csv",
        "description": "Test dataset",
        "owner_id": "the owner",
        "created_at": _CREATED_AT,
        "examples": [example_1, example_2],
    }
    mock_post.return_value = mock_response

    client = LangChainClient(api_url="http://localhost:8000", api_key="123")
    csv_file = ("test.csv", BytesIO(b"input,output\n1,2\n3,4\n"))

    dataset = client.upload_csv(
        csv_file, "Test dataset", input_keys=["input"], output_keys=["output"]
    )

    assert dataset.id == "1"
    assert dataset.name == "test.csv"
    assert dataset.description == "Test dataset"
    assert dataset.examples == [example_1, example_2]
