import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_models.hunyuan import ChatHunyuan


@pytest.mark.requires("tencentcloud-sdk-python")
def test_chat_hunyuan() -> None:
    chat = ChatHunyuan()
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.requires("tencentcloud-sdk-python")
def test_chat_hunyuan_with_temperature() -> None:
    chat = ChatHunyuan(temperature=0.6)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.requires("tencentcloud-sdk-python")
def test_chat_hunyuan_with_model_name() -> None:
    chat = ChatHunyuan(model="hunyuan-standard")
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.requires("tencentcloud-sdk-python")
def test_chat_hunyuan_with_stream() -> None:
    chat = ChatHunyuan(streaming=True)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_extra_kwargs() -> None:
    chat = ChatHunyuan(temperature=0.88, top_p=0.7)
    assert chat.temperature == 0.88
    assert chat.top_p == 0.7
