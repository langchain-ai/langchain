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


def test_on_llm_end_finetuned_model(handler: OpenAICallbackHandler) -> None:
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
    handler.on_llm_end(response)
    assert handler.total_cost > 0
