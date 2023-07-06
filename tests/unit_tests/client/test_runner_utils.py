"""Test the LangChain+ client."""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from unittest import mock

import pytest
from langchainplus_sdk.client import LangChainPlusClient
from langchainplus_sdk.schemas import Dataset, Example

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.transform import TransformChain
from langchain.client.runner_utils import (
    InputFormatError,
    _get_messages,
    _get_prompts,
    arun_on_dataset,
    run_llm,
    run_llm_or_chain,
)
from langchain.schema import LLMResult
from tests.unit_tests.llms.fake_chat_model import FakeChatModel
from tests.unit_tests.llms.fake_llm import FakeLLM

_CREATED_AT = datetime(2015, 1, 1, 0, 0, 0)
_TENANT_ID = "7a3d2b56-cd5b-44e5-846f-7eb6e8144ce4"
_EXAMPLE_MESSAGE = {
    "data": {"content": "Foo", "example": False, "additional_kwargs": {}},
    "type": "human",
}
_VALID_MESSAGES = [
    {"messages": [_EXAMPLE_MESSAGE], "other_key": "value"},
    {"messages": [], "other_key": "value"},
    {
        "messages": [[_EXAMPLE_MESSAGE, _EXAMPLE_MESSAGE], [_EXAMPLE_MESSAGE]],
        "other_key": "value",
    },
    {"any_key": [_EXAMPLE_MESSAGE]},
    {"any_key": [[_EXAMPLE_MESSAGE, _EXAMPLE_MESSAGE], [_EXAMPLE_MESSAGE]]},
]
_VALID_PROMPTS = [
    {"prompts": ["foo", "bar", "baz"], "other_key": "value"},
    {"prompt": "foo", "other_key": ["bar", "baz"]},
    {"some_key": "foo"},
    {"some_key": ["foo", "bar"]},
]


@pytest.mark.parametrize(
    "inputs",
    _VALID_MESSAGES,
)
def test__get_messages_valid(inputs: Dict[str, Any]) -> None:
    {"messages": []}
    _get_messages(inputs)


@pytest.mark.parametrize(
    "inputs",
    _VALID_PROMPTS,
)
def test__get_prompts_valid(inputs: Dict[str, Any]) -> None:
    _get_prompts(inputs)


@pytest.mark.parametrize(
    "inputs",
    [
        {"prompts": "foo"},
        {"prompt": ["foo"]},
        {"some_key": 3},
        {"some_key": "foo", "other_key": "bar"},
    ],
)
def test__get_prompts_invalid(inputs: Dict[str, Any]) -> None:
    with pytest.raises(InputFormatError):
        _get_prompts(inputs)


def test_run_llm_or_chain_with_input_mapper() -> None:
    example = Example(
        id=uuid.uuid4(),
        created_at=_CREATED_AT,
        inputs={"the wrong input": "1", "another key": "2"},
        outputs={"output": "2"},
        dataset_id=str(uuid.uuid4()),
    )

    def run_val(inputs: dict) -> dict:
        assert "the right input" in inputs
        return {"output": "2"}

    mock_chain = TransformChain(
        input_variables=["the right input"],
        output_variables=["output"],
        transform=run_val,
    )

    def input_mapper(inputs: dict) -> dict:
        assert "the wrong input" in inputs
        return {"the right input": inputs["the wrong input"]}

    result = run_llm_or_chain(
        example, lambda: mock_chain, n_repetitions=1, input_mapper=input_mapper
    )
    assert len(result) == 1
    assert result[0] == {"output": "2", "the right input": "1"}
    bad_result = run_llm_or_chain(
        example,
        lambda: mock_chain,
        n_repetitions=1,
    )
    assert len(bad_result) == 1
    assert "Error" in bad_result[0]

    # Try with LLM
    def llm_input_mapper(inputs: dict) -> List[str]:
        assert "the wrong input" in inputs
        return ["the right input"]

    mock_llm = FakeLLM(queries={"the right input": "somenumber"})
    result = run_llm_or_chain(
        example, mock_llm, n_repetitions=1, input_mapper=llm_input_mapper
    )
    assert len(result) == 1
    llm_result = result[0]
    assert isinstance(llm_result, LLMResult)
    assert llm_result.generations[0][0].text == "somenumber"


@pytest.mark.parametrize(
    "inputs",
    [
        {"one_key": [_EXAMPLE_MESSAGE], "other_key": "value"},
        {
            "messages": [[_EXAMPLE_MESSAGE, _EXAMPLE_MESSAGE], _EXAMPLE_MESSAGE],
            "other_key": "value",
        },
        {"prompts": "foo"},
        {},
    ],
)
def test__get_messages_invalid(inputs: Dict[str, Any]) -> None:
    with pytest.raises(InputFormatError):
        _get_messages(inputs)


@pytest.mark.parametrize("inputs", _VALID_PROMPTS + _VALID_MESSAGES)
def test_run_llm_all_formats(inputs: Dict[str, Any]) -> None:
    llm = FakeLLM()
    run_llm(llm, inputs, mock.MagicMock())


@pytest.mark.parametrize("inputs", _VALID_MESSAGES + _VALID_PROMPTS)
def test_run_chat_model_all_formats(inputs: Dict[str, Any]) -> None:
    llm = FakeChatModel()
    run_llm(llm, inputs, mock.MagicMock())


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
        tags: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        return [
            {"result": f"Result for example {example.id}"} for _ in range(n_repetitions)
        ]

    def mock_create_project(*args: Any, **kwargs: Any) -> None:
        pass

    with mock.patch.object(
        LangChainPlusClient, "read_dataset", new=mock_read_dataset
    ), mock.patch.object(
        LangChainPlusClient, "list_examples", new=mock_list_examples
    ), mock.patch(
        "langchain.client.runner_utils._arun_llm_or_chain", new=mock_arun_chain
    ), mock.patch.object(
        LangChainPlusClient, "create_project", new=mock_create_project
    ):
        client = LangChainPlusClient(api_url="http://localhost:1984", api_key="123")
        chain = mock.MagicMock()
        num_repetitions = 3
        results = await arun_on_dataset(
            dataset_name="test",
            llm_or_chain_factory=lambda: chain,
            concurrency_level=2,
            project_name="test_project",
            num_repetitions=num_repetitions,
            client=client,
        )

        expected = {
            uuid_: [
                {"result": f"Result for example {uuid.UUID(uuid_)}"}
                for _ in range(num_repetitions)
            ]
            for uuid_ in uuids
        }
        assert results["results"] == expected
