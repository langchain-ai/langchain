"""Test AI21 Chat API wrapper."""
from unittest.mock import Mock, call

import pytest
from ai21 import MissingApiKeyError
from ai21.models import (
    Penalty,
)

from langchain_ai21 import AI21LLM
from tests.unit_tests.conftest import (
    BASIC_EXAMPLE_LLM_PARAMETERS,
    DUMMY_API_KEY,
    temporarily_unset_api_key,
)


def test_initialization__when_no_api_key__should_raise_exception() -> None:
    """Test integration initialization."""
    with temporarily_unset_api_key():
        with pytest.raises(MissingApiKeyError):
            AI21LLM(
                model="j2-ultra",
            )


def test_initialization__when_default_parameters() -> None:
    """Test integration initialization."""
    AI21LLM(
        api_key=DUMMY_API_KEY,
        model="j2-ultra",
    )


def test_initialization__when_custom_parameters_to_init() -> None:
    """Test integration initialization."""
    AI21LLM(
        api_key=DUMMY_API_KEY,
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


def test_generate(mock_client_with_completion: Mock) -> None:
    # Setup test
    prompt0 = "Hi, my name is what?"
    prompt1 = "My name is who?"
    stop = ["\n"]
    custom_model = "test_model"
    epoch = 1

    ai21 = AI21LLM(
        model="j2-ultra",
        api_key=DUMMY_API_KEY,
        client=mock_client_with_completion,
        custom_model=custom_model,
        epoch=epoch,
        **BASIC_EXAMPLE_LLM_PARAMETERS,
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
                custom_model=custom_model,
                stop_sequences=stop,
                epoch=epoch,
                **BASIC_EXAMPLE_LLM_PARAMETERS,
            ),
            call(
                prompt=prompt1,
                model="j2-ultra",
                custom_model=custom_model,
                stop_sequences=stop,
                epoch=epoch,
                **BASIC_EXAMPLE_LLM_PARAMETERS,
            ),
        ]
    )
