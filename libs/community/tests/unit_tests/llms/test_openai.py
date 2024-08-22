import pytest

from langchain_community.llms.openai import OpenAI
from langchain_community.utils.openai import is_openai_v1


def _openai_v1_installed() -> bool:
    try:
        return is_openai_v1()
    except Exception as _:
        return False


@pytest.mark.requires("openai")
def test_openai_model_param() -> None:
    llm = OpenAI(model="foo", openai_api_key="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"
    llm = OpenAI(model_name="foo", openai_api_key="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"


@pytest.mark.requires("openai")
def test_openai_model_kwargs() -> None:
    llm = OpenAI(model_kwargs={"foo": "bar"}, openai_api_key="foo")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("openai")
def test_openai_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        OpenAI(model_kwargs={"model_name": "foo"})

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        OpenAI(model_kwargs={"model": "gpt-3.5-turbo-instruct"})


@pytest.mark.requires("openai")
def test_openai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OpenAI(foo="bar", openai_api_key="foo")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "cmpl-3evkmQda5Hu7fcZavknQda3SQ",
        "object": "text_completion",
        "created": 1689989000,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [
            {"text": "Bar Baz", "index": 0, "logprobs": None, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
