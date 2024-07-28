import os
from typing import List

import pytest

from langchain_neospace import NeoSpace

os.environ["NEOSPACE_API_KEY"] = "foo"


def test_neospace_model_param() -> None:
    llm = NeoSpace(model="foo")
    assert llm.model_name == "foo"
    llm = NeoSpace(model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"


def test_neospace_model_kwargs() -> None:
    llm = NeoSpace(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


def test_neospace_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        NeoSpace(model_kwargs={"model_name": "foo"})


def test_neospace_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = NeoSpace(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "cmpl-3evkmQda5Hu7fcZavknQda3SQ",
        "object": "text_completion",
        "created": 1689989000,
        "model": "text-davinci-003",
        "choices": [
            {"text": "Bar Baz", "index": 0, "logprobs": None, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


@pytest.mark.parametrize("model", ["7b-math-rank16", "text-davinci-003"])
def test_get_token_ids(model: str) -> None:
    NeoSpace(model=model).get_token_ids("foo")
    return


def test_custom_token_counting() -> None:
    def token_encoder(text: str) -> List[int]:
        return [1, 2, 3]

    llm = NeoSpace(custom_get_token_ids=token_encoder)
    assert llm.get_token_ids("foo") == [1, 2, 3]
