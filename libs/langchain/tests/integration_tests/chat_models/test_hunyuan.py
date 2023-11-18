from langchain.chat_models.hunyuan import ChatHunyuan
from langchain.schema.messages import AIMessage, HumanMessage


def test_chat_hunyuan() -> None:
    chat = ChatHunyuan()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_hunyuan_with_temperature() -> None:
    chat = ChatHunyuan(temperature=0.6)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_extra_kwargs() -> None:
    chat = ChatHunyuan(temperature=0.88, top_p=0.7)
    assert chat.temperature == 0.88
    assert chat.top_p == 0.7
