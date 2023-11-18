from langchain.chat_models.baichuan import ChatBaichuan
from langchain.schema.messages import AIMessage, HumanMessage


def test_chat_baichuan() -> None:
    chat = ChatBaichuan()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_with_model() -> None:
    chat = ChatBaichuan(model="Baichuan2-13B")
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_with_temperature() -> None:
    chat = ChatBaichuan(model="Baichuan2-13B", temperature=1.0)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_with_kwargs() -> None:
    chat = ChatBaichuan()
    message = HumanMessage(content="Hello")
    response = chat([message], temperature=0.88, top_p=0.7)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_extra_kwargs() -> None:
    chat = ChatBaichuan(temperature=0.88, top_p=0.7)
    assert chat.temperature == 0.88
    assert chat.top_p == 0.7
