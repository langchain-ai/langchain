"""Test Vertex AI API wrapper."""

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

from langchain.chat_models.vertexai import ChatVertexAI
from langchain.schema import (
    HumanMessage,
)

if TYPE_CHECKING:
    import pytest


def test_vertexai_args_passed(monkeypatch: "pytest.MonkeyPatch") -> None:
    # Mock/suppress credential validation in langchain
    monkeypatch.setattr(ChatVertexAI, "validate_environment", Mock())

    response_text = "Goodbye"
    user_prompt = "Hello"
    prompt_params = {
        "max_output_tokens": 1,
        "temperature": 10000.0,
        "top_k": 10,
        "top_p": 0.5,
    }

    # Mock the library to ensure the args are passed correctly
    with patch(
        "vertexai.language_models._language_models.ChatSession.send_message"
    ) as send_message:
        response = Mock(text=response_text)
        send_message.return_value = response

        model = ChatVertexAI(**prompt_params)
        message = HumanMessage(content=user_prompt)
        response = model([message])

        assert response.content == response_text
        send_message.assert_called_once_with(
            user_prompt,
            **prompt_params,
        )
