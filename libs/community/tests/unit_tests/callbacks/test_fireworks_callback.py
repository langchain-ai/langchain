from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.outputs import LLMResult

from langchain_community.callbacks import FireworksCallbackHandler
from langchain_community.llms.fireworks import Fireworks


@pytest.fixture
def handler() -> FireworksCallbackHandler:
    return FireworksCallbackHandler()


def test_on_llm_end_single_choice(handler: FireworksCallbackHandler()) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": [
                {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
            ],
            "model": Fireworks.__fields__["model"].default,
        },
    )
    handler.on_llm_end(response)
    assert handler.successful_requests == 1
    assert handler.total_tokens == 3
    assert handler.completion_tokens == 1
    assert handler.prompt_tokens == 2
    assert handler.total_cost > 0


def test_on_llm_end_multiple_choices(handler: FireworksCallbackHandler) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": [
                {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
            ],
            "model": Fireworks.__fields__["model"].default,
        },
    )
    handler.on_llm_end(response)
    assert handler.successful_requests == 1
    assert handler.total_tokens == 9
    assert handler.completion_tokens == 5
    assert handler.prompt_tokens == 4
    assert handler.total_cost > 0


@pytest.mark.parametrize(
    "model, expected_cost",
    [
        ("accounts/fireworks/models/mistral-7b-instruct-4k", 0.001),
        ("accounts/fireworks/models/llama-v2-34b-code-instruct", 0.0035),
        ("accounts/fireworks/models/mixtral-8x7b", 0.002),
    ],
)
def test_on_llm_end_single_choice_cost(
    handler: FireworksCallbackHandler, model: str, expected_cost: float
) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": [
                {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000}
            ],
            "model": model,
        },
    )
    handler.on_llm_end(response)
    assert handler.total_cost == expected_cost


@pytest.mark.parametrize(
    "model, expected_cost",
    [
        ("accounts/fireworks/models/mistral-7b-instruct-4k", 0.001),
        ("accounts/fireworks/models/llama-v2-34b-code-instruct", 0.0035),
        ("accounts/fireworks/models/mixtral-8x7b", 0.002),
    ],
)
def test_on_llm_end_multiple_choice_cost(
    handler: FireworksCallbackHandler, model: str, expected_cost: float
) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "token_usage": [
                {"prompt_tokens": 500, "completion_tokens": 500, "total_tokens": 1000},
                {"prompt_tokens": 500, "completion_tokens": 500, "total_tokens": 1000},
            ],
            "model": model,
        },
    )
    handler.on_llm_end(response)
    assert handler.total_cost == expected_cost


def test_on_retry_works(handler: FireworksCallbackHandler) -> None:
    handler.on_retry(MagicMock(), run_id=uuid4())
