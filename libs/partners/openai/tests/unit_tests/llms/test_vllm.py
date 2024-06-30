import os
from typing import List

import pytest

from langchain_openai import VLLMOpenAI

os.environ["OPENAI_API_KEY"] = "foo"


def test_vllm_model_param() -> None:
    llm = VLLMOpenAI(model="foo")
    assert llm.model_name == "foo"
    llm = VLLMOpenAI(model_name="foo")
    assert llm.model_name == "foo"


def test_vllm_model_kwargs() -> None:
    llm = VLLMOpenAI(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


def test_vllm_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        VLLMOpenAI(model_kwargs={"model_name": "foo"})


def test_vllm_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = VLLMOpenAI(foo="bar")
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


@pytest.mark.parametrize("model", ["gpt-3.5-turbo-instruct", "text-davinci-003"])
def test_get_token_ids(model: str) -> None:
    VLLMOpenAI(model=model).get_token_ids("foo")
    return


def test_custom_token_counting() -> None:
    def token_encoder(text: str) -> List[int]:
        return [1, 2, 3]

    llm = VLLMOpenAI(custom_get_token_ids=token_encoder)
    assert llm.get_token_ids("foo") == [1, 2, 3]
