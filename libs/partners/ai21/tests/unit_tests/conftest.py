import os
from contextlib import contextmanager
from typing import Generator
from unittest.mock import Mock

import pytest
from ai21 import AI21Client
from ai21.models import (
    ChatOutput,
    ChatResponse,
    Completion,
    CompletionData,
    CompletionFinishReason,
    CompletionsResponse,
    FinishReason,
    Penalty,
    RoleType,
)
from pytest_mock import MockerFixture

DUMMY_API_KEY = "test_api_key"


BASIC_EXAMPLE_LLM_PARAMETERS = {
    "num_results": 3,
    "max_tokens": 20,
    "min_tokens": 10,
    "temperature": 0.5,
    "top_p": 0.5,
    "top_k_return": 0,
    "frequency_penalty": Penalty(scale=0.2, apply_to_numbers=True),
    "presence_penalty": Penalty(scale=0.2, apply_to_stopwords=True),
    "count_penalty": Penalty(
        scale=0.2, apply_to_punctuation=True, apply_to_emojis=True
    ),
}


@pytest.fixture
def mocked_completion_response(mocker: MockerFixture) -> Mock:
    mocked_response = mocker.MagicMock(spec=CompletionsResponse)
    mocked_response.prompt = "this is a test prompt"
    mocked_response.completions = [
        Completion(
            data=CompletionData(text="test", tokens=[]),
            finish_reason=CompletionFinishReason(reason=None, length=None),
        )
    ]
    return mocked_response


@pytest.fixture
def mock_client_with_completion(
    mocker: MockerFixture, mocked_completion_response: Mock
) -> Mock:
    mock_client = mocker.MagicMock(spec=AI21Client)
    mock_client.completion = mocker.MagicMock()
    mock_client.completion.create.side_effect = [
        mocked_completion_response,
        mocked_completion_response,
    ]
    mock_client.count_tokens.side_effect = [10, 20]

    return mock_client


@pytest.fixture
def mock_client_with_chat(mocker: MockerFixture) -> Mock:
    mock_client = mocker.MagicMock(spec=AI21Client)
    mock_client.chat = mocker.MagicMock()

    output = ChatOutput(
        text="Hello Pickle Rick!",
        role=RoleType.ASSISTANT,
        finish_reason=FinishReason(reason="testing"),
    )
    mock_client.chat.create.return_value = ChatResponse(outputs=[output])

    return mock_client


@contextmanager
def temporarily_unset_api_key() -> Generator:
    """
    Unset and set environment key for testing purpose for when an API KEY is not set
    """
    api_key = os.environ.pop("API_KEY", None)
    yield

    if api_key is not None:
        os.environ["API_KEY"] = api_key
