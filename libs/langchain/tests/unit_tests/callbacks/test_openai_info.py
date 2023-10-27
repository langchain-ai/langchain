from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from langchain.callbacks import OpenAICallbackHandler
from langchain.llms.openai import BaseOpenAI
from langchain.schema import LLMResult


@pytest.fixture
def handler() -> OpenAICallbackHandler:
    return OpenAICallbackHandler()


def test_on_llm_end(handler: OpenAICallbackHandler) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
            "model_name": BaseOpenAI.__fields__["model_name"].default,
        },
    )
    handler.on_llm_end(response)
    assert handler.successful_requests == 1
    assert handler.total_tokens == 3
    assert handler.prompt_tokens == 2
    assert handler.completion_tokens == 1
    assert handler.total_cost > 0


def test_on_llm_end_custom_model(handler: OpenAICallbackHandler) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
            "model_name": "foo-bar",
        },
    )
    handler.on_llm_end(response)
    assert handler.total_cost == 0


@pytest.mark.parametrize(
    "model_name",
    [
        "ada:ft-your-org:custom-model-name-2022-02-15-04-21-04",
        "babbage:ft-your-org:custom-model-name-2022-02-15-04-21-04",
        "curie:ft-your-org:custom-model-name-2022-02-15-04-21-04",
        "davinci:ft-your-org:custom-model-name-2022-02-15-04-21-04",
        "ft:babbage-002:your-org:custom-model-name:1abcdefg",
        "ft:davinci-002:your-org:custom-model-name:1abcdefg",
        "ft:gpt-3.5-turbo-0613:your-org:custom-model-name:1abcdefg",
        "babbage-002.ft-0123456789abcdefghijklmnopqrstuv",
        "davinci-002.ft-0123456789abcdefghijklmnopqrstuv",
        "gpt-35-turbo-0613.ft-0123456789abcdefghijklmnopqrstuv",
    ],
)
def test_on_llm_end_finetuned_model(
    handler: OpenAICallbackHandler, model_name: str
) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
            "model_name": model_name,
        },
    )
    handler.on_llm_end(response)
    assert handler.total_cost > 0


@pytest.mark.parametrize(
    "model_name,expected_cost",
    [
        ("gpt-35-turbo", 0.0035),
        ("gpt-35-turbo-0301", 0.0035),
        (
            "gpt-35-turbo-0613",
            0.0035,
        ),
        (
            "gpt-35-turbo-16k-0613",
            0.007,
        ),
        (
            "gpt-35-turbo-16k",
            0.007,
        ),
        ("gpt-4", 0.09),
        ("gpt-4-0314", 0.09),
        ("gpt-4-0613", 0.09),
        ("gpt-4-32k", 0.18),
        ("gpt-4-32k-0314", 0.18),
        ("gpt-4-32k-0613", 0.18),
    ],
)
def test_on_llm_end_azure_openai(
    handler: OpenAICallbackHandler, model_name: str, expected_cost: float
) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "total_tokens": 2000,
            },
            "model_name": model_name,
        },
    )
    handler.on_llm_end(response)
    assert handler.total_cost == expected_cost


@pytest.mark.parametrize(
    "model_name", ["gpt-35-turbo-16k-0301", "gpt-4-0301", "gpt-4-32k-0301"]
)
def test_on_llm_end_no_cost_invalid_model(
    handler: OpenAICallbackHandler, model_name: str
) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "total_tokens": 2000,
            },
            "model_name": model_name,
        },
    )
    handler.on_llm_end(response)
    assert handler.total_cost == 0


def test_on_retry_works(handler: OpenAICallbackHandler) -> None:
    handler.on_retry(MagicMock(), run_id=uuid4())
