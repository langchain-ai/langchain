"""Test Vertex AI API wrapper.
In order to run this test, you need to install VertexAI SDK 
pip install google-cloud-aiplatform>=1.25.0

Your end-user credentials would be used to make the calls (make sure you've run 
`gcloud auth login` first).
"""
from unittest.mock import MagicMock, Mock, patch

import pytest

from langchain.chat_models import ChatVertexAI
from langchain.chat_models.vertexai import (
    TextChatVertexAI,
    _history_to_str,
    _parse_chat_history,
    _parse_examples,
)
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage


def test_vertexai_single_call() -> None:
    model = ChatVertexAI()
    message = HumanMessage(content="Hello")
    response = model([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert model._llm_type == "vertexai"
    assert model.model_name == model.client._model_id


def test_vertexai_single_call_with_context() -> None:
    model = ChatVertexAI()
    raw_context = (
        "My name is Ned. You are my personal assistant. My favorite movies "
        "are Lord of the Rings and Hobbit."
    )
    question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    context = SystemMessage(content=raw_context)
    message = HumanMessage(content=question)
    response = model([context, message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_vertexai_single_call_with_examples() -> None:
    model = ChatVertexAI()
    raw_context = "My name is Ned. You are my personal assistant."
    question = "2+2"
    text_question, text_answer = "4+4", "8"
    inp = HumanMessage(content=text_question)
    output = AIMessage(content=text_answer)
    context = SystemMessage(content=raw_context)
    message = HumanMessage(content=question)
    response = model([context, message], examples=[inp, output])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_parse_chat_history_correct() -> None:
    from vertexai.language_models import ChatMessage

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


def test_vertexai_single_call_failes_no_message() -> None:
    chat = ChatVertexAI()
    with pytest.raises(ValueError) as exc_info:
        _ = chat([])
    assert (
        str(exc_info.value)
        == "You should provide at least one message to start the chat!"
    )


def test_vertexai_args_passed() -> None:
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
        "vertexai.language_models._language_models.ChatModel.start_chat"
    ) as start_chat:
        mock_response = Mock(text=response_text)
        mock_chat = MagicMock()
        start_chat.return_value = mock_chat
        mock_send_message = MagicMock(return_value=mock_response)
        mock_chat.send_message = mock_send_message

        model = ChatVertexAI(**prompt_params)
        message = HumanMessage(content=user_prompt)
        response = model([message])

        assert response.content == response_text
        mock_send_message.assert_called_once_with(user_prompt)
        start_chat.assert_called_once_with(
            context=None, message_history=[], **prompt_params
        )


def test_parse_examples_correct() -> None:
    from vertexai.language_models import InputOutputTextPair

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


def test_parse_exmaples_failes_wrong_sequence() -> None:
    with pytest.raises(ValueError) as exc_info:
        _ = _parse_examples([AIMessage(content="a")])
    assert (
        str(exc_info.value)
        == "Expect examples to have an even amount of messages, got 1."
    )


def test_history_to_str() -> None:
    text_question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    question = HumanMessage(content=text_question)
    text_answer = (
        "Sure, You might enjoy The Lord of the Rings: The Fellowship of the Ring "
        "(2001): This is the first movie in the Lord of the Rings trilogy."
    )
    answer = AIMessage(content=text_answer)
    history = _history_to_str([question, answer])
    assert history == f"Input: {text_question}\nOutput: {text_answer}"


def test_textchatvertexai() -> None:
    model = TextChatVertexAI()
    raw_context = (
        "My name is Ned. You are my personal assistant. "
        "Given the {history}, continue the dialogue."
    )
    text_question2 = "2+2"
    text_question1, text_answer1 = "4+4", "8"
    question1 = HumanMessage(content=text_question1)
    answer1 = AIMessage(content=text_answer1)
    context = SystemMessage(content=raw_context)
    question2 = HumanMessage(content=text_question2)
    response = model([context, question1, answer1, question2])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_textchatvertexai_call_args() -> None:
    import vertexai

    with patch.object(
        vertexai.language_models._language_models.TextGenerationModel, "from_pretrained"
    ) as mock_text_model:
        mock_predict = MagicMock()
        result = MagicMock()
        result.text = "some_result"
        mock_predict.return_value = result
        mock_model = MagicMock()
        mock_text_model.return_value = mock_model
        mock_model.predict = mock_predict

        model = TextChatVertexAI()
        raw_context = (
            "My name is Ned. You are my personal assistant. "
            "Given the {history}, continue the dialogue."
        )
        text_question2 = "2+2"
        text_question1, text_answer1 = "4+4", "8"
        question1 = HumanMessage(content=text_question1)
        answer1 = AIMessage(content=text_answer1)
        context = SystemMessage(content=raw_context)
        question2 = HumanMessage(content=text_question2)
        _ = model([context, question1, answer1, question2])
        expected_prompt = (
            "My name is Ned. You are my personal assistant. "
            "Given the Input: 4+4\nOutput: 8, continue the dialogue."
        )
        mock_predict.assert_called_once_with(
            expected_prompt,
            temperature=0.0,
            max_output_tokens=128,
            top_k=40,
            top_p=0.95,
        )
