import pytest

from langchain.chat_models.ernie import ErnieBotChat
from langchain.schema.messages import AIMessage, HumanMessage


def test_chat_ernie_bot() -> None:
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_ernie_bot_with_model_name() -> None:
    chat = ErnieBotChat(model_name="ERNIE-Bot")
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_ernie_bot_with_temperature() -> None:
    chat = ErnieBotChat(model_name="ERNIE-Bot", temperature=1.0)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_ernie_bot_with_kwargs() -> None:
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    response = chat([message], temperature=0.88, top_p=0.7)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_extra_kwargs() -> None:
    chat = ErnieBotChat(temperature=0.88, top_p=0.7)
    assert chat.temperature == 0.88
    assert chat.top_p == 0.7


def test_wrong_temperature_1() -> None:
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    with pytest.raises(ValueError):
        chat([message], temperature=1.2)


def test_wrong_temperature_2() -> None:
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    with pytest.raises(ValueError):
        chat([message], temperature=0)
