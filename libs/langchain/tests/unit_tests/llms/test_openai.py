import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain.llms.openai import OpenAI

os.environ["OPENAI_API_KEY"] = "foo"


@pytest.mark.requires("openai")
def test_openai_model_param() -> None:
    llm = OpenAI(model="foo")
    assert llm.model_name == "foo"
    llm = OpenAI(model_name="foo")
    assert llm.model_name == "foo"


@pytest.mark.requires("openai")
def test_openai_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        OpenAI(model_kwargs={"model_name": "foo"})


@pytest.mark.requires("openai")
def test_openai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OpenAI(foo="bar")
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


@pytest.mark.requires("openai")
def test_openai_calls(mock_completion: dict) -> None:
    llm = OpenAI()
    mock_client = MagicMock()
    completed = False

    def raise_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = raise_once
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.predict("bar")
        assert res == "Bar Baz"
    assert completed


@pytest.mark.requires("openai")
async def test_openai_async_retries(mock_completion: dict) -> None:
    llm = OpenAI()
    mock_client = MagicMock()
    completed = False

    def raise_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = raise_once
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.apredict("bar")
        assert res == "Bar Baz"
    assert completed
