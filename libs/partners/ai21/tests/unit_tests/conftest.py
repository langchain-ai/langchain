import os
from contextlib import contextmanager
from typing import Generator
from unittest.mock import Mock

import pytest
from ai21 import AI21Client, AI21EnvConfig
from ai21.models import (
    AnswerResponse,
    ChatOutput,
    ChatResponse,
    Completion,
    CompletionData,
    CompletionFinishReason,
    CompletionsResponse,
    FinishReason,
    Penalty,
    RoleType,
    SegmentationResponse,
)
from ai21.models.responses.segmentation_response import Segment
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
        scale=0.2,
        apply_to_punctuation=True,
        apply_to_emojis=True,
    ),
}

SEGMENTS = [
    Segment(
        segment_type="normal_text",
        segment_text=(
            "The original full name of the franchise is Pocket Monsters "
            "(ポケットモンスター, Poketto Monsutā), which was abbreviated to "
            "Pokemon during development of the original games.\n\nWhen the "
            "franchise was released internationally, the short form of the "
            "title was used, with an acute accent (´) over the e to aid "
            "in pronunciation."
        ),
    ),
    Segment(
        segment_type="normal_text",
        segment_text=(
            "Pokémon refers to both the franchise itself and the creatures "
            "within its fictional universe.\n\nAs a noun, it is identical in "
            "both the singular and plural, as is every individual species "
            'name;[10] it is grammatically correct to say "one Pokémon" '
            'and "many Pokémon", as well as "one Pikachu" and "many '
            'Pikachu".\n\nIn English, Pokémon may be pronounced either '
            "/'powkɛmon/ (poe-keh-mon) or /'powkɪmon/ (poe-key-mon)."
        ),
    ),
]


BASIC_EXAMPLE_LLM_PARAMETERS_AS_DICT = {
    "num_results": 3,
    "max_tokens": 20,
    "min_tokens": 10,
    "temperature": 0.5,
    "top_p": 0.5,
    "top_k_return": 0,
    "frequency_penalty": Penalty(scale=0.2, apply_to_numbers=True).to_dict(),
    "presence_penalty": Penalty(scale=0.2, apply_to_stopwords=True).to_dict(),
    "count_penalty": Penalty(
        scale=0.2,
        apply_to_punctuation=True,
        apply_to_emojis=True,
    ).to_dict(),
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
    api_key = AI21EnvConfig.api_key
    AI21EnvConfig.api_key = None
    os.environ.pop("AI21_API_KEY", None)
    yield

    if api_key is not None:
        AI21EnvConfig.api_key = api_key
        os.environ["AI21_API_KEY"] = api_key


@pytest.fixture
def mock_client_with_contextual_answers(mocker: MockerFixture) -> Mock:
    mock_client = mocker.MagicMock(spec=AI21Client)
    mock_client.answer = mocker.MagicMock()
    mock_client.answer.create.return_value = AnswerResponse(
        id="some_id",
        answer="some answer",
        answer_in_context=False,
    )

    return mock_client


@pytest.fixture
def mock_client_with_semantic_text_splitter(mocker: MockerFixture) -> Mock:
    mock_client = mocker.MagicMock(spec=AI21Client)
    mock_client.segmentation = mocker.MagicMock()
    mock_client.segmentation.create.return_value = SegmentationResponse(
        id="12345",
        segments=SEGMENTS,
    )

    return mock_client
