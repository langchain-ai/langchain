from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.outputs import LLMResult

from langchain_community.callbacks import BedrockTokenUsageCallbackHandler


@pytest.fixture
def handler() -> BedrockTokenUsageCallbackHandler:
    return BedrockTokenUsageCallbackHandler()


def test_on_llm_end(handler: BedrockTokenUsageCallbackHandler) -> None:
    response = LLMResult(
        generations=[],
        llm_output={
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        },
    )
    handler.on_llm_end(response)
    assert handler.successful_requests == 1
    assert handler.total_tokens == 3
    assert handler.prompt_tokens == 2
    assert handler.completion_tokens == 1

    handler.on_llm_end(response)
    assert handler.successful_requests == 1 * 2
    assert handler.total_tokens == 3 * 2
    assert handler.prompt_tokens == 2 * 2
    assert handler.completion_tokens == 1 * 2
