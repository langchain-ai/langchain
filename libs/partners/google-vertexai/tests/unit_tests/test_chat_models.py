"""Test chat model integration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from vertexai.language_models import ChatMessage, InputOutputTextPair  # type: ignore

from langchain_google_vertexai.chat_models import (
    ChatVertexAI,
    _parse_chat_history,
    _parse_chat_history_gemini,
    _parse_examples,
)


def test_parse_examples_correct() -> None:
    text_question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    question = HumanMessage(content=text_question)
    text_answer = (
        "Sure, You might enjoy The Lord of the Rings: The Fellowship of the Ring "
        "(2001): This is the first movie in the Lord of the Rings trilogy."
    )
    answer = AIMessage(content=text_answer)
    examples = _parse_examples([question, answer, question, answer])
    assert len(examples) == 2
    assert examples == [
        InputOutputTextPair(input_text=text_question, output_text=text_answer),
        InputOutputTextPair(input_text=text_question, output_text=text_answer),
    ]


def test_parse_examples_failes_wrong_sequence() -> None:
    with pytest.raises(ValueError) as exc_info:
        _ = _parse_examples([AIMessage(content="a")])
    assert (
        str(exc_info.value)
        == "Expect examples to have an even amount of messages, got 1."
    )


@dataclass
class StubTextChatResponse:
    """Stub text-chat response from VertexAI for testing."""

    text: str


@pytest.mark.parametrize("stop", [None, "stop1"])
def test_vertexai_args_passed(stop: Optional[str]) -> None:
    response_text = "Goodbye"
    user_prompt = "Hello"
    prompt_params: Dict[str, Any] = {
        "max_output_tokens": 1,
        "temperature": 10000.0,
        "top_k": 10,
        "top_p": 0.5,
    }

    # Mock the library to ensure the args are passed correctly
    with patch("vertexai._model_garden._model_garden_models._from_pretrained") as mg:
        mock_response = MagicMock()
        mock_response.candidates = [StubTextChatResponse(text=response_text)]
        mock_chat = MagicMock()
        mock_send_message = MagicMock(return_value=mock_response)
        mock_chat.send_message = mock_send_message

        mock_model = MagicMock()
        mock_start_chat = MagicMock(return_value=mock_chat)
        mock_model.start_chat = mock_start_chat
        mg.return_value = mock_model

        model = ChatVertexAI(**prompt_params)
        message = HumanMessage(content=user_prompt)
        if stop:
            response = model([message], stop=[stop])
        else:
            response = model([message])

        assert response.content == response_text
        mock_send_message.assert_called_once_with(user_prompt, candidate_count=1)
        expected_stop_sequence = [stop] if stop else None
        mock_start_chat.assert_called_once_with(
            context=None,
            message_history=[],
            **prompt_params,
            stop_sequences=expected_stop_sequence,
        )


def test_parse_chat_history_correct() -> None:
    text_context = (
        "My name is Ned. You are my personal assistant. My "
        "favorite movies are Lord of the Rings and Hobbit."
    )
    context = SystemMessage(content=text_context)
    text_question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    question = HumanMessage(content=text_question)
    text_answer = (
        "Sure, You might enjoy The Lord of the Rings: The Fellowship of the Ring "
        "(2001): This is the first movie in the Lord of the Rings trilogy."
    )
    answer = AIMessage(content=text_answer)
    history = _parse_chat_history([context, question, answer, question, answer])
    assert history.context == context.content
    assert len(history.history) == 4
    assert history.history == [
        ChatMessage(content=text_question, author="user"),
        ChatMessage(content=text_answer, author="bot"),
        ChatMessage(content=text_question, author="user"),
        ChatMessage(content=text_answer, author="bot"),
    ]


def test_parse_history_gemini() -> None:
    system_input = "You're supposed to answer math questions."
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    system_message = SystemMessage(content=system_input)
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    messages = [system_message, message1, message2, message3]
    history = _parse_chat_history_gemini(messages, convert_system_message_to_human=True)
    assert len(history) == 3
    assert history[0].role == "user"
    assert history[0].parts[0].text == system_input
    assert history[0].parts[1].text == text_question1
    assert history[1].role == "model"
    assert history[1].parts[0].text == text_answer1


def test_default_params_palm() -> None:
    user_prompt = "Hello"

    with patch("vertexai._model_garden._model_garden_models._from_pretrained") as mg:
        mock_response = MagicMock()
        mock_response.candidates = [StubTextChatResponse(text="Goodbye")]
        mock_chat = MagicMock()
        mock_send_message = MagicMock(return_value=mock_response)
        mock_chat.send_message = mock_send_message

        mock_model = MagicMock()
        mock_start_chat = MagicMock(return_value=mock_chat)
        mock_model.start_chat = mock_start_chat
        mg.return_value = mock_model

        model = ChatVertexAI(model_name="text-bison@001")
        message = HumanMessage(content=user_prompt)
        _ = model([message])
        mock_start_chat.assert_called_once_with(
            context=None,
            message_history=[],
            max_output_tokens=128,
            top_k=40,
            top_p=0.95,
            stop_sequences=None,
        )


@dataclass
class StubGeminiResponse:
    """Stub gemini response from VertexAI for testing."""

    text: str
    content: Any
    citation_metadata: Any
    safety_ratings: List[Any] = field(default_factory=list)


def test_default_params_gemini() -> None:
    user_prompt = "Hello"

    with patch("langchain_google_vertexai.chat_models.GenerativeModel") as gm:
        mock_response = MagicMock()
        mock_response.candidates = [
            StubGeminiResponse(
                text="Goodbye",
                content=Mock(parts=[Mock(function_call=None)]),
                citation_metadata=Mock(),
            )
        ]
        mock_chat = MagicMock()
        mock_send_message = MagicMock(return_value=mock_response)
        mock_chat.send_message = mock_send_message

        mock_model = MagicMock()
        mock_start_chat = MagicMock(return_value=mock_chat)
        mock_model.start_chat = mock_start_chat
        gm.return_value = mock_model
        model = ChatVertexAI(model_name="gemini-pro")
        message = HumanMessage(content=user_prompt)
        _ = model([message])
        mock_start_chat.assert_called_once_with(history=[])
