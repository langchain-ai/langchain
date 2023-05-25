"""Test the LangChain+ client."""
from typing import Any, Dict
from unittest import mock

import pytest

from langchain.client.runner_utils import (
    InputFormatError,
    _get_messages,
    _get_prompts,
    run_llm,
)
from tests.unit_tests.llms.fake_chat_model import FakeChatModel
from tests.unit_tests.llms.fake_llm import FakeLLM

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
