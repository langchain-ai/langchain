from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_models.baichuan import ChatBaichuan

# For testing, run:
# TEST_FILE=tests/integration_tests/chat_models/test_baichuan.py make test


def test_chat_baichuan_default() -> None:
    chat = ChatBaichuan(streaming=True)
    message = HumanMessage(content="请完整背诵将进酒，背诵5遍")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_default_non_streaming() -> None:
    chat = ChatBaichuan()
    message = HumanMessage(content="请完整背诵将进酒，背诵5遍")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_turbo() -> None:
    chat = ChatBaichuan(model="Baichuan2-Turbo", streaming=True)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_turbo_non_streaming() -> None:
    chat = ChatBaichuan(model="Baichuan2-Turbo")
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_with_temperature() -> None:
    chat = ChatBaichuan(temperature=1.0)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_with_kwargs() -> None:
    chat = ChatBaichuan()
    message = HumanMessage(content="百川192K API是什么时候上线的？")
    response = chat([message], temperature=0.88, top_p=0.7, with_search_enhance=True)
    print(response)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_extra_kwargs() -> None:
    chat = ChatBaichuan(temperature=0.88, top_p=0.7, with_search_enhance=True)
    assert chat.temperature == 0.88
    assert chat.top_p == 0.7
    assert chat.with_search_enhance is True
