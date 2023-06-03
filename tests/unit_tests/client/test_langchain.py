"""Test the LangChain+ client."""
import uuid
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Union
from unittest import mock

import pytest

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.client.langchain import (
    LangChainPlusClient,
    _get_link_stem,
    _is_localhost,
)
from langchain.client.models import Dataset, Example

_CREATED_AT = datetime(2015, 1, 1, 0, 0, 0)
_TENANT_ID = "7a3d2b56-cd5b-44e5-846f-7eb6e8144ce4"


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


def test_validate_api_key_if_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key must be provided"):
        LangChainPlusClient(api_url="http://www.example.com")

    client = LangChainPlusClient(api_url="http://localhost:8000")
    assert client.api_url == "http://localhost:8000"
    assert client.api_key is None


def test_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    client = LangChainPlusClient(api_url="http://localhost:8000", api_key="123")
    assert client._headers == {"x-api-key": "123"}

    client_no_key = LangChainPlusClient(api_url="http://localhost:8000")
    assert client_no_key._headers == {}


@mock.patch("langchain.client.langchain.requests.post")
def test_upload_csv(mock_post: mock.Mock) -> None:
    mock_response = mock.Mock()
    dataset_id = str(uuid.uuid4())
    example_1 = Example(
        id=str(uuid.uuid4()),
        created_at=_CREATED_AT,
        inputs={"input": "1"},
        outputs={"output": "2"},
        dataset_id=dataset_id,
    )
    example_2 = Example(
        id=str(uuid.uuid4()),
        created_at=_CREATED_AT,
        inputs={"input": "3"},
        outputs={"output": "4"},
        dataset_id=dataset_id,
    )

    mock_response.json.return_value = {
        "id": dataset_id,
        "name": "test.csv",
        "description": "Test dataset",
        "owner_id": "the owner",
        "created_at": _CREATED_AT,
        "examples": [example_1, example_2],
        "tenant_id": _TENANT_ID,
    }
    mock_post.return_value = mock_response

    client = LangChainPlusClient(
        api_url="http://localhost:8000",
        api_key="123",
    )
    csv_file = ("test.csv", BytesIO(b"input,output\n1,2\n3,4\n"))

    dataset = client.upload_csv(
        csv_file, "Test dataset", input_keys=["input"], output_keys=["output"]
    )

    assert dataset.id == uuid.UUID(dataset_id)
    assert dataset.name == "test.csv"
    assert dataset.description == "Test dataset"


@pytest.mark.asyncio
async def test_arun_on_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset(
        id=uuid.uuid4(),
        name="test",
        description="Test dataset",
        owner_id="owner",
        created_at=_CREATED_AT,
        tenant_id=_TENANT_ID,
    )
    uuids = [
        "0c193153-2309-4704-9a47-17aee4fb25c8",
        "0d11b5fd-8e66-4485-b696-4b55155c0c05",
        "90d696f0-f10d-4fd0-b88b-bfee6df08b84",
        "4ce2c6d8-5124-4c0c-8292-db7bdebcf167",
        "7b5a524c-80fa-4960-888e-7d380f9a11ee",
    ]
    examples = [
        Example(
            id=uuids[0],
            created_at=_CREATED_AT,
            inputs={"input": "1"},
            outputs={"output": "2"},
            dataset_id=str(uuid.uuid4()),
        ),
        Example(
            id=uuids[1],
            created_at=_CREATED_AT,
            inputs={"input": "3"},
            outputs={"output": "4"},
            dataset_id=str(uuid.uuid4()),
        ),
        Example(
            id=uuids[2],
            created_at=_CREATED_AT,
            inputs={"input": "5"},
            outputs={"output": "6"},
            dataset_id=str(uuid.uuid4()),
        ),
        Example(
            id=uuids[3],
            created_at=_CREATED_AT,
            inputs={"input": "7"},
            outputs={"output": "8"},
            dataset_id=str(uuid.uuid4()),
        ),
        Example(
            id=uuids[4],
            created_at=_CREATED_AT,
            inputs={"input": "9"},
            outputs={"output": "10"},
            dataset_id=str(uuid.uuid4()),
        ),
    ]

    def mock_read_dataset(*args: Any, **kwargs: Any) -> Dataset:
        return dataset

    def mock_list_examples(*args: Any, **kwargs: Any) -> List[Example]:
        return examples

    async def mock_arun_chain(
        example: Example,
        llm_or_chain: Union[BaseLanguageModel, Chain],
        n_repetitions: int,
        tracer: Any,
    ) -> List[Dict[str, Any]]:
        return [
            {"result": f"Result for example {example.id}"} for _ in range(n_repetitions)
        ]

    with mock.patch.object(
        LangChainPlusClient, "read_dataset", new=mock_read_dataset
    ), mock.patch.object(
        LangChainPlusClient, "list_examples", new=mock_list_examples
    ), mock.patch(
        "langchain.client.runner_utils._arun_llm_or_chain", new=mock_arun_chain
    ):
        client = LangChainPlusClient(api_url="http://localhost:8000", api_key="123")
        chain = mock.MagicMock()
        num_repetitions = 3
        results = await client.arun_on_dataset(
            dataset_name="test",
            llm_or_chain_factory=lambda: chain,
            concurrency_level=2,
            session_name="test_session",
            num_repetitions=num_repetitions,
        )

        expected = {
            uuid_: [
                {"result": f"Result for example {uuid.UUID(uuid_)}"}
                for _ in range(num_repetitions)
            ]
            for uuid_ in uuids
        }
        assert results == expected
