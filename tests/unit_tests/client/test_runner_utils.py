"""Test the LangChain+ client."""
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from unittest import mock

import pytest
from langchainplus_sdk.client import LangChainPlusClient
from langchainplus_sdk.schemas import Dataset, Example

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.client.runner_utils import (
    InputFormatError,
    _get_messages,
    _get_prompts,
    arun_on_dataset,
    run_llm,
)
from tests.unit_tests.chains.test_base import FakeChain
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


@pytest.fixture
def dataset() -> Dataset:
    return Dataset(
        id=uuid.uuid4(),
        name="test",
        description="Test dataset",
        owner_id="owner",
        created_at=_CREATED_AT,
        tenant_id=_TENANT_ID,
    )


@pytest.fixture
def uuids() -> List[str]:
    return [
        "0c193153-2309-4704-9a47-17aee4fb25c8",
        "0d11b5fd-8e66-4485-b696-4b55155c0c05",
        "90d696f0-f10d-4fd0-b88b-bfee6df08b84",
        "4ce2c6d8-5124-4c0c-8292-db7bdebcf167",
        "7b5a524c-80fa-4960-888e-7d380f9a11ee",
    ]


@pytest.fixture
def examples(uuids: List[str]) -> List[Example]:
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
    return examples


_RUN_OBJECTS = [
    FakeLLM(
        queres={str(i): f"Result for input {i}" for i in range(10)},
        sequential_responses=True,
    ),
    FakeChatModel(),
    lambda: FakeChain(the_input_keys=["input"], the_output_keys=["output"]),
    lambda input_: {"result": f"Result for input {input_}"},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", _RUN_OBJECTS)
async def test_arun_on_dataset(
    monkeypatch: pytest.MonkeyPatch,
    examples: List[Example],
    dataset: Dataset,
    uuids: List[str],
    model: Union[BaseLanguageModel, Chain, Callable[[dict], dict]],
) -> None:
    def mock_read_dataset(*args: Any, **kwargs: Any) -> Dataset:
        return dataset

    def mock_list_examples(*args: Any, **kwargs: Any) -> List[Example]:
        return examples

    def mock_create_project(*args: Any, **kwargs: Any) -> None:
        pass

    with mock.patch.object(
        LangChainPlusClient, "read_dataset", new=mock_read_dataset
    ), mock.patch.object(
        LangChainPlusClient, "list_examples", new=mock_list_examples
    ), mock.patch.object(
        LangChainPlusClient, "create_project", new=mock_create_project
    ), mock.patch(
        "langchain.client.runner_utils.LangChainTracer", mock.MagicMock
    ):
        client = LangChainPlusClient(api_url="http://localhost:1984", api_key="123")
        num_repetitions = 3
        results = await arun_on_dataset(
            dataset_name="test",
            llm_or_chain_factory=model,
            concurrency_level=2,
            project_name="test_project",
            num_repetitions=num_repetitions,
            client=client,
        )
        assert "results" in results
        assert results["results"]
        # expected = {
        #     uuid_: [
        #         {"result": f"Result for example {uuid.UUID(uuid_)}"}
        #         for _ in range(num_repetitions)
        #     ]
        #     for uuid_ in uuids
        # }
        # assert results["results"] == expected
