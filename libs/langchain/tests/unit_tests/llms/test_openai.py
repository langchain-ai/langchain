import asyncio
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain.llms.openai import OpenAI
from tests.unit_tests.callbacks.fake_callback_handler import (
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)

os.environ["OPENAI_API_KEY"] = "foo"


@pytest.mark.requires("openai")
def test_openai_model_param() -> None:
    llm = OpenAI(model="foo")
    assert llm.model_name == "foo"
    llm = OpenAI(model_name="foo")
    assert llm.model_name == "foo"


@pytest.mark.requires("openai")
def test_openai_model_kwargs() -> None:
    llm = OpenAI(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


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
def test_openai_retries(mock_completion: dict) -> None:
    llm = OpenAI()
    mock_client = MagicMock()
    completed = False
    raised = False
    import openai

    def raise_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed, raised
        if not raised:
            raised = True
            raise openai.error.APIError
        completed = True
        return mock_completion

    mock_client.create = raise_once
    callback_handler = FakeCallbackHandler()
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.predict("bar", callbacks=[callback_handler])
        assert res == "Bar Baz"
    assert completed
    assert raised
    assert callback_handler.retries == 1


@pytest.mark.requires("openai")
@pytest.mark.asyncio
async def test_openai_async_retries(mock_completion: dict) -> None:
    llm = OpenAI()
    mock_client = MagicMock()
    completed = False
    raised = False
    import openai

    async def araise_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed, raised
        if not raised:
            raised = True
            raise openai.error.APIError
        await asyncio.sleep(0)
        completed = True
        return mock_completion

    mock_client.acreate = araise_once
    callback_handler = FakeAsyncCallbackHandler()
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = await llm.apredict("bar", callbacks=[callback_handler])
        assert res == "Bar Baz"
    assert completed
    assert raised
    assert callback_handler.retries == 1
