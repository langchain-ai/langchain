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
