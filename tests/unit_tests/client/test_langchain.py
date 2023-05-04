"""Test the LangChain+ client."""
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List
from unittest import mock

import pytest

from langchain.chains.base import Chain
from langchain.client.langchain import LangChainClient, _get_link_stem, _is_localhost
from langchain.client.models import Dataset, Example

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


@pytest.mark.asyncio
async def test_arun_chain_on_dataset() -> None:
    dataset = Dataset(
        id="1",
        name="test",
        description="Test dataset",
        owner_id="owner",
        created_at=_CREATED_AT,
    )
    examples = [
        Example(
            id="1",
            created_at=_CREATED_AT,
            inputs={"input": "1"},
            outputs={"output": "2"},
            dataset_id="1",
        ),
        Example(
            id="2",
            created_at=_CREATED_AT,
            inputs={"input": "3"},
            outputs={"output": "4"},
            dataset_id="1",
        ),
        Example(
            id="3",
            created_at=_CREATED_AT,
            inputs={"input": "5"},
            outputs={"output": "6"},
            dataset_id="1",
        ),
        Example(
            id="4",
            created_at=_CREATED_AT,
            inputs={"input": "7"},
            outputs={"output": "8"},
            dataset_id="1",
        ),
        Example(
            id="5",
            created_at=_CREATED_AT,
            inputs={"input": "9"},
            outputs={"output": "10"},
            dataset_id="1",
        ),
    ]

    async def mock_aread_dataset(*args: Any, **kwargs: Any) -> Dataset:
        return dataset

    async def mock_alist_examples(*args: Any, **kwargs: Any) -> List[Example]:
        return examples

    async def mock_arun_chain(
        example: Example, tracer: Any, chain: Chain
    ) -> Dict[str, Any]:
        return {"result": f"Result for example {example.id}"}

    with mock.patch.object(
        LangChainClient, "aread_dataset", new=mock_aread_dataset
    ), mock.patch.object(
        LangChainClient, "alist_examples", new=mock_alist_examples
    ), mock.patch.object(
        LangChainClient, "_arun_chain", new=mock_arun_chain
    ):
        client = LangChainClient(api_url="http://localhost:8000", api_key="123")
        chain = mock.MagicMock()

        results = await client.arun_chain_on_dataset(
            dataset_name="test", chain=chain, num_workers=2, session_name="test_session"
        )

        assert results == {
            "1": {"result": "Result for example 1"},
            "2": {"result": "Result for example 2"},
            "3": {"result": "Result for example 3"},
            "4": {"result": "Result for example 4"},
            "5": {"result": "Result for example 5"},
        }
