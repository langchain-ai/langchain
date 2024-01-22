"""Test AI21 Chat API wrapper."""
import os
from unittest.mock import Mock, call

import pytest
from ai21 import AI21Client
from ai21.models import (
    Penalty,
    CompletionsResponse,
    Completion,
    CompletionData,
    CompletionFinishReason,
)
from pytest_mock import MockerFixture

from langchain_ai21 import AI21

os.environ["AI21_API_KEY"] = "test_key"


@pytest.fixture
def mocked_completion_response(mocker: MockerFixture):
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
    mocker: MockerFixture, mocked_completion_response
) -> Mock:
    mock_client = mocker.MagicMock(spec=AI21Client)
    mock_client.completion = mocker.MagicMock()
    mock_client.completion.create.side_effect = [
        mocked_completion_response,
        mocked_completion_response,
    ]
    mock_client.count_tokens.side_effect = [10, 20]

    return mock_client


def test_initialization__when_default_parameters() -> None:
    """Test integration initialization."""
    AI21()


def test_initialization__when_custom_parameters_to_init() -> None:
    """Test integration initialization."""
    AI21(
        model="j2-mid",
        num_results=2,
        max_tokens=20,
        min_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k_returns=0,
        stop_sequences=["\n"],
        frequency_penalty=Penalty(scale=0.2, apply_to_numbers=True),
        presence_penalty=Penalty(scale=0.2, apply_to_stopwords=True),
        count_penalty=Penalty(
            scale=0.2, apply_to_punctuation=True, apply_to_emojis=True
        ),
        custom_model="test_model",
        epoch=1,
    )


def test_generate(mock_client_with_completion):
    # Setup test
    prompt0 = "Hi, my name is what?"
    prompt1 = "My name is who?"
    stop = ["\n"]
    num_results = 3
    max_tokens = 20
    min_tokens = 10
    temperature = 0.5
    top_p = 0.5
    top_k_returns = 0
    frequency_penalty = Penalty(scale=0.2, apply_to_numbers=True)
    presence_penalty = Penalty(scale=0.2, apply_to_stopwords=True)
    count_penalty = Penalty(scale=0.2, apply_to_punctuation=True, apply_to_emojis=True)
    custom_model = "test_model"
    epoch = 1

    ai21 = AI21(
        client=mock_client_with_completion,
        num_results=num_results,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k_return=top_k_returns,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        count_penalty=count_penalty,
        custom_model=custom_model,
        epoch=epoch,
    )

    # Make call to testing function
    ai21.generate(
        [prompt0, prompt1],
        stop=stop,
    )

    # Assertions
    mock_client_with_completion.count_tokens.assert_has_calls(
        [
            call(prompt0),
            call(prompt1),
        ],
    )

    mock_client_with_completion.completion.create.assert_has_calls(
        [
            call(
                prompt=prompt0,
                model="j2-ultra",
                max_tokens=max_tokens,
                num_results=num_results,
                min_tokens=min_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k_return=top_k_returns,
                custom_model=custom_model,
                stop_sequences=stop,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                count_penalty=count_penalty,
                epoch=epoch,
            ),
            call(
                prompt=prompt1,
                model="j2-ultra",
                max_tokens=max_tokens,
                num_results=num_results,
                min_tokens=min_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k_return=top_k_returns,
                custom_model=custom_model,
                stop_sequences=stop,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                count_penalty=count_penalty,
                epoch=epoch,
            ),
        ]
    )
