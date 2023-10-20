from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from langchain.callbacks import OpenAICallbackHandler
from langchain.llms.openai import BaseOpenAI
from langchain.schema import LLMResult


@pytest.fixture
def handler() -> OpenAICallbackHandler:
    return OpenAICallbackHandler()


@pytest.fixture
def run_id() -> UUID:
    return uuid4()


def test_on_llm_end(handler: OpenAICallbackHandler, run_id: UUID) -> None:
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
    handler.on_llm_end(response, run_id=run_id)
    assert handler.successful_requests == 1
    assert handler.total_tokens == 3
    assert handler.prompt_tokens == 2
    assert handler.completion_tokens == 1
    assert handler.total_cost > 0


def test_on_llm_end_custom_model(handler: OpenAICallbackHandler, run_id: UUID) -> None:
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
    handler.on_llm_end(response, run_id=run_id)
    assert handler.total_cost == 0


def test_on_llm_end_finetuned_model(
    handler: OpenAICallbackHandler, run_id: UUID
) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
            "model_name": "ada:ft-your-org:custom-model-name-2022-02-15-04-21-04",
        },
    )
    handler.on_llm_end(response, run_id=run_id)
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
    handler: OpenAICallbackHandler, model_name: str, expected_cost: float, run_id: UUID
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
    handler.on_llm_end(response, run_id=run_id)
    assert handler.total_cost == expected_cost


@pytest.mark.parametrize(
    "model_name", ["gpt-35-turbo-16k-0301", "gpt-4-0301", "gpt-4-32k-0301"]
)
def test_on_llm_end_no_cost_invalid_model(
    handler: OpenAICallbackHandler, model_name: str, run_id: UUID
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
    handler.on_llm_end(response, run_id=run_id)
    assert handler.total_cost == 0


def test_on_retry_works(handler: OpenAICallbackHandler, run_id: UUID) -> None:
    handler.on_retry(MagicMock(), run_id=run_id)


def test_on_llm_end_multiple_models(
    handler: OpenAICallbackHandler, run_id: UUID
) -> None:
    models = ["gpt-3.5-turbo", "gpt-4"]
    for model in models:
        llm_result = LLMResult(
            generations=[],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
                "model_name": model,
            },
        )
        handler.on_llm_end(llm_result, run_id=run_id)
    assert handler.total_cost == 0.000125
    assert handler.prompt_tokens == 4
    assert handler.completion_tokens == 2
    assert handler.total_tokens == 6
    assert handler.successful_requests == 2


def test_on_new_token(handler: OpenAICallbackHandler, run_id: UUID) -> None:
    handler.current_model_name[run_id] = "gpt-3.5-turbo"
    handler.on_llm_new_token("", run_id=run_id)
    for _ in range(10):
        handler.on_llm_new_token(token="a", run_id=run_id)
    handler.on_llm_new_token("", run_id=run_id)
    handler.on_llm_end(
        LLMResult(generations=[]),
        run_id=run_id,
    )
    assert handler.total_cost == 2e-05
    assert handler.prompt_tokens == 0
    assert handler.completion_tokens == 10
    assert handler.total_tokens == 10
    assert handler.successful_requests == 0


def test_on_new_token_multiple_models(handler: OpenAICallbackHandler) -> None:
    models = ["gpt-3.5-turbo", "gpt-4"]
    for model in models:
        run_id = uuid4()
        handler.current_model_name[run_id] = model
        handler.on_llm_new_token("", run_id=run_id)
        for _ in range(10):
            handler.on_llm_new_token(token="a", run_id=run_id)
        handler.on_llm_new_token("", run_id=run_id)
        handler.on_llm_end(
            LLMResult(generations=[]),
            run_id=run_id,
        )
    assert handler.total_cost == 0.00062
    assert handler.prompt_tokens == 0
    assert handler.completion_tokens == 20
    assert handler.total_tokens == 20
    assert handler.successful_requests == 0
