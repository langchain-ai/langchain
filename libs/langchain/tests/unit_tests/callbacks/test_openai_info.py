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

@pytest.mark.parametrize(
        "model_name,expected_cost", 
        [
            ("gpt-35-turbo", 0.004), # 0.002 per 1k tokens both input and output
            ("gpt-35-turbo-0301", 0.004), # 0.002 per 1k tokens both input and output
            ("gpt-35-turbo-0613", 0.0035), # 0.0015 per 1k input tokens; 0.002 per 1k output tokens
            ("gpt-35-turbo-16k-0613", 0.007), # 0.003 per 1k input tokens; 0.004 per 1k output tokens
            ("gpt-35-turbo-16k", 0.007), # 0.003 per 1k input tokens; 0.004 per 1k output tokens
            ("gpt-4", 0.09), # 0,003 per 1k input tokens; 0.006 per 1k output tokens
            ("gpt-4-0314", 0.09), # 0,03 per 1k input tokens; 0.06 per 1k output tokens
            ("gpt-4-0613", 0.09), # 0,03 per 1k input tokens; 0.06 per 1k output tokens
            ("gpt-4-32k", 0.18), # 0,06 per 1k input tokens; 0.12 per 1k output tokens
            ("gpt-4-32k-0314", 0.18), # 0,06 per 1k input tokens; 0.12 per 1k output tokens
            ("gpt-4-32k-0613", 0.18), # 0,06 per 1k input tokens; 0.12 per 1k output tokens
        ]
    )
def test_on_llm_end_azure_openai(handler: OpenAICallbackHandler, model_name: str, expected_cost: float) -> None:
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

@pytest.mark.parametrize("model_name", ["gpt-35-turbo-16k-0301", "gpt-4-0301", "gpt-4-32k-0301"])
def test_on_llm_end_no_cost_invalid_model(handler: OpenAICallbackHandler, model_name: str) -> None:
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