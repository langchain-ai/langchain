"""Test Vertex AI API wrapper.
In order to run this test, you need to install VertexAI SDK (that is is the private
preview)  and be whitelisted to list the models themselves:
export SDK=google_cloud_aiplatform-1.25.dev20230413+language.models-py2.py3-none-any.whl
gsutil cp gs://vertex_sdk_llm_private_releases/SDK/${SDK SDK} .
pip install invoke
pip install ${SDK} "shapely<2.0.0"
Your end-user credentials would be used to make the calls (make sure you've run 
`gcloud auth login` first).
"""
import pytest

from langchain.chat_models import ChatVertexAI
from langchain.chat_models.vertexai import _MessagePair, _parse_chat_history
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


def test_vertexai_single_call() -> None:
    model = ChatVertexAI()
    message = HumanMessage(content="Hello")
    response = model([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert model.chat._context is None
    assert model._llm_type == "vertexai"
    assert len(model.history) == 2
    assert model.history[0] == HumanMessage(content="Hello")
    assert model.history[1] == response
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
    assert model.chat._context == raw_context


def test_vertexai_single_call_restart_chat() -> None:
    model = ChatVertexAI()
    raw_context = (
        "My name is Ned. You are my personal assistant. My favorite movies "
        "are Lord of the Rings and Hobbit."
    )
    question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    response = model(
        [SystemMessage(content=raw_context), HumanMessage(content=question)]
    )
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert model.chat._context == raw_context
    question2 = (
        "Hello, could you recommend a good movie for me to watch this weekend, please?"
    )
    response2 = model([HumanMessage(content=question2)])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert model.chat._context is None
    assert len(model.history) == 2
    assert model.history[0] == HumanMessage(content=question2)
    assert model.history[1] == response2


def test_vertexai_send_message() -> None:
    model = ChatVertexAI()
    raw_context = "My name is Ned. You are my personal assistant."
    response = model(
        [SystemMessage(content=raw_context), HumanMessage(content="Hello")]
    )
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert model.chat._context == raw_context
    assert len(model.history) == 2
    model.send_message("Tell me a joke.")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(model.history) == 4
    assert model.chat._context == raw_context


def test_vertexai_single_call_with_context_and_history() -> None:
    model = ChatVertexAI()
    raw_context = (
        "My name is Ned. You are my personal assistant. My favorite movies "
        "are Lord of the Rings and Hobbit."
    )
    question1 = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    answer = (
        "Sure, You might enjoy The Lord of the Rings: The Fellowship of the Ring "
        "(2001): This is the first movie in the Lord of the Rings trilogy."
    )
    question2 = "Thank you, I saw this one already. Can you recommend me another one?"
    context = SystemMessage(content=raw_context)
    response = model(
        [
            context,
            HumanMessage(content=question1),
            AIMessage(content=answer),
            HumanMessage(content=question2),
        ]
    )
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert model.chat._context == raw_context
    assert len(model.history) == 4
    assert model.history[0] == HumanMessage(content=question1)
    assert model.history[1] == AIMessage(content=answer)
    assert model.history[2] == HumanMessage(content=question2)


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
    assert history.system_message == context
    assert len(history.history) == 2
    assert history.history[0] == _MessagePair(question=question, answer=answer)


def test_parse_chat_history_wrong_sequence() -> None:
    text_question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    question = HumanMessage(content=text_question)
    with pytest.raises(ValueError) as exc_info:
        _ = _parse_chat_history([question, question])
    assert (
        str(exc_info.value)
        == "A human message should follow a bot one, got human, human."
    )


def test_vertexai_single_call_failes_no_message() -> None:
    chat = ChatVertexAI()
    with pytest.raises(ValueError) as exc_info:
        _ = chat([])
    assert (
        str(exc_info.value)
        == "You should provide at least one message to start the chat!"
    )
