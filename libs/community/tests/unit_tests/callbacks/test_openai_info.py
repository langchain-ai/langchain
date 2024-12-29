from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.utils.pydantic import get_fields

from langchain_community.callbacks import OpenAICallbackHandler
from langchain_community.llms.openai import BaseOpenAI


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
            "model_name": get_fields(BaseOpenAI)["model_name"].default,
        },
    )
    handler.on_llm_end(response)
    assert handler.successful_requests == 1
    assert handler.total_tokens == 3
    assert handler.prompt_tokens == 2
    assert handler.completion_tokens == 1
    assert handler.total_cost > 0


def test_on_llm_end_with_chat_generation(handler: OpenAICallbackHandler) -> None:
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    text="Hello, world!",
                    message=AIMessage(
                        content="Hello, world!",
                        usage_metadata={
                            "input_tokens": 2,
                            "output_tokens": 2,
                            "total_tokens": 4,
                            "input_token_details": {
                                "cache_read": 1,
                            },
                            "output_token_details": {
                                "reasoning": 1,
                            },
                        },
                    ),
                )
            ]
        ],
        llm_output={
            "model_name": get_fields(BaseOpenAI)["model_name"].default,
        },
    )
    handler.on_llm_end(response)
    assert handler.successful_requests == 1
    assert handler.total_tokens == 4
    assert handler.prompt_tokens == 2
    assert handler.prompt_tokens_cached == 1
    assert handler.completion_tokens == 2
    assert handler.reasoning_tokens == 1
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
    "model_name, expected_cost",
    [
        ("ada:ft-your-org:custom-model-name-2022-02-15-04-21-04", 0.0032),
        ("babbage:ft-your-org:custom-model-name-2022-02-15-04-21-04", 0.0048),
        ("curie:ft-your-org:custom-model-name-2022-02-15-04-21-04", 0.024),
        ("davinci:ft-your-org:custom-model-name-2022-02-15-04-21-04", 0.24),
        ("ft:babbage-002:your-org:custom-model-name:1abcdefg", 0.0032),
        ("ft:davinci-002:your-org:custom-model-name:1abcdefg", 0.024),
        ("ft:gpt-3.5-turbo-0613:your-org:custom-model-name:1abcdefg", 0.009),
        ("babbage-002.ft-0123456789abcdefghijklmnopqrstuv", 0.0008),
        ("davinci-002.ft-0123456789abcdefghijklmnopqrstuv", 0.004),
        ("gpt-35-turbo-0613.ft-0123456789abcdefghijklmnopqrstuv", 0.0035),
    ],
)
def test_on_llm_end_finetuned_model(
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
    assert np.isclose(handler.total_cost, expected_cost)


@pytest.mark.parametrize(
    "model_name,expected_cost",
    [
        ("gpt-35-turbo", 0.0035),
        ("gpt-35-turbo-0301", 0.004),
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
