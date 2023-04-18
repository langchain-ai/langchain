"""Test Vertex AI API wrapper.

In order to run this test, you need to install VertexAI SDK (that is is the private preview) and be whitelisted to list the models themselves:

gsutil cp gs://vertex_sdk_llm_private_releases/SDK/google_cloud_aiplatform-1.25.dev20230413+language.models-py2.py3-none-any.whl .
pip install invoke
pip install google_cloud_aiplatform-1.25.dev20230413+language.models-py2.py3-none-any.whl "shapely<2.0.0"

Your end-user credentials would be used to make the calls (make sure you've run `gcloud auth login` first).
"""
from typing import List

import pytest

from langchain.chat_models import ChatVertexAI, MultiTurnChatVertexAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


def test_vertexai_single_call() -> None:
    chat = ChatVertexAI()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert chat._llm_type == "chat-bison-001"


def test_vertexai_single_call_with_context() -> None:
    chat = ChatVertexAI()
    context = SystemMessage(
        content="My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit."
    )
    message = HumanMessage(
        content="Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    response = chat([context, message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_vertexai_single_call_failes_wrong_amount_of_messages() -> None:
    chat = ChatVertexAI()
    context = SystemMessage(
        content="My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit."
    )
    message = HumanMessage(
        content="Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    with pytest.raises(ValueError) as exc_info:
        _ = chat([context, message, message])
    assert str(exc_info.value) == "Chat model expects only one or two messages!"


def test_vertexai_single_call_failes_no_human_message() -> None:
    chat = ChatVertexAI()
    context = SystemMessage(
        content="My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit."
    )
    with pytest.raises(ValueError) as exc_info:
        _ = chat([context])
    assert str(exc_info.value) == "Message should be from human if it's the first one!"


def test_vertexai_single_call_failes_two_human_messages() -> None:
    chat = ChatVertexAI()
    message = HumanMessage(
        content="Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    with pytest.raises(ValueError) as exc_info:
        _ = chat([message, message])
    assert (
        str(exc_info.value)
        == "The first message should be a system one if there're two of them."
    )


def test_vertexai_single_call_failes_two_ai_messages() -> None:
    chat = ChatVertexAI()
    context = SystemMessage(
        content="My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit."
    )
    with pytest.raises(ValueError) as exc_info:
        _ = chat([context, context])
    assert str(exc_info.value) == "The second message should from human!"


def test_vertexai_multiturn_chat_with_context() -> None:
    chat = MultiTurnChatVertexAI()
    context = SystemMessage(
        content="My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit."
    )
    message1 = HumanMessage(
        content="Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    message2 = HumanMessage(
        content="I've already seen this movie, could you recommend another one, please?"
    )
    chat.start_chat(context)
    turn1 = chat([message1])
    assert isinstance(turn1, AIMessage)
    assert isinstance(turn1.content, str)
    assert len(chat.history) == 1
    assert chat.history[0] == (message1.content, turn1.content)
    assert chat.chat._context == context.content  # type: ignore
    turn2 = chat([message2])
    assert isinstance(turn2, AIMessage)
    assert isinstance(turn2.content, str)
    assert len(chat.history) == 2
    assert chat.history[1] == (message2.content, turn2.content)
    assert chat._llm_type == "chat-bison-001"


def test_vertexai_multiturn_chat_without_context() -> None:
    chat = MultiTurnChatVertexAI()
    message1 = HumanMessage(
        content="Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    message2 = HumanMessage(
        content="I've already seen this movie, could you recommend another one, please?"
    )
    chat.start_chat()
    turn1 = chat([message1])
    assert isinstance(turn1, AIMessage)
    assert isinstance(turn1.content, str)
    assert len(chat.history) == 1
    assert chat.history[0] == (message1.content, turn1.content)
    assert chat.chat._context is None  # type: ignore
    turn2 = chat([message2])
    assert isinstance(turn2, AIMessage)
    assert isinstance(turn2.content, str)
    assert len(chat.history) == 2
    assert chat.history[1] == (message2.content, turn2.content)


def test_vertexai_multiturn_chat_requires_start_chat() -> None:
    chat = MultiTurnChatVertexAI()
    message = HumanMessage(
        content="Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    with pytest.raises(ValueError) as exc_info:
        _ = chat([message])
    assert str(exc_info.value) == "You should start_chat first!"


def test_vertexai_multiturn_chat_clear_chat() -> None:
    chat = MultiTurnChatVertexAI()
    message = HumanMessage(
        content="Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    chat.start_chat()
    _ = chat([message])
    chat.clear_chat()
    assert chat.chat is None
    chat.start_chat()
    assert chat.chat
