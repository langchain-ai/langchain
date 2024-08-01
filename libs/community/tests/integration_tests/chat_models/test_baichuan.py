from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_models.baichuan import ChatBaichuan

# For testing, run:
# TEST_FILE=tests/integration_tests/chat_models/test_baichuan.py make test


def test_chat_baichuan_default() -> None:
    chat = ChatBaichuan(streaming=True)  # type: ignore[call-arg]
    message = HumanMessage(content="请完整背诵将进酒，背诵5遍")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_default_non_streaming() -> None:
    chat = ChatBaichuan()  # type: ignore[call-arg]
    message = HumanMessage(content="请完整背诵将进酒，背诵5遍")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_turbo() -> None:
    chat = ChatBaichuan(model="Baichuan2-Turbo", streaming=True)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_turbo_non_streaming() -> None:
    chat = ChatBaichuan(model="Baichuan2-Turbo")  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_with_temperature() -> None:
    chat = ChatBaichuan(temperature=1.0)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_baichuan_with_kwargs() -> None:
    chat = ChatBaichuan()  # type: ignore[call-arg]
    message = HumanMessage(content="百川192K API是什么时候上线的？")
    response = chat.invoke(
        [message], temperature=0.88, top_p=0.7, with_search_enhance=True
    )
    print(response)  # noqa: T201
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_extra_kwargs() -> None:
    chat = ChatBaichuan(temperature=0.88, top_p=0.7, with_search_enhance=True)  # type: ignore[call-arg]
    assert chat.temperature == 0.88
    assert chat.top_p == 0.7
    assert chat.with_search_enhance is True


async def test_chat_baichuan_agenerate() -> None:
    chat = ChatBaichuan()  # type: ignore[call-arg]
    response = await chat.ainvoke("你好呀")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


async def test_chat_baichuan_astream() -> None:
    chat = ChatBaichuan()  # type: ignore[call-arg]
    async for chunk in chat.astream("今天天气如何？"):
        assert isinstance(chunk, AIMessage)
        assert isinstance(chunk.content, str)


def test_chat_baichuan_with_system_role() -> None:
    chat = ChatBaichuan()  # type: ignore[call-arg]
    messages = [
        ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
        ("human", "我喜欢编程。"),
    ]
    response = chat.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
