from typing import cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain.chat_models.baichuan import ChatBaichuan


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


def test_baichuan_key_is_secret_string() -> None:
    chat = ChatBaichuan()
    assert isinstance(chat.baichuan_api_key, SecretStr)
    assert isinstance(chat.baichuan_secret_key, SecretStr)


def test_baichuan_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("BAICHUAN_API_KEY", "test-api-key")
    monkeypatch.setenv("BAICHUAN_SECRET_KEY", "test-secret-key")

    chat = ChatBaichuan()
    print(chat.baichuan_api_key, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"

    print(chat.baichuan_secret_key, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"


def test_baichuan_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    chat = ChatBaichuan(
        baichuan_api_key="test-api-key", baichuan_secret_key="test-secret-key"
    )
    print(chat.baichuan_api_key, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"

    print(chat.baichuan_secret_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secret_str() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    chat = ChatBaichuan(
        baichuan_api_key="test-api-key", baichuan_secret_key="test-secret-key"
    )
    assert cast(SecretStr, chat.baichuan_api_key).get_secret_value() == "test-api-key"
    assert (
        cast(SecretStr, chat.baichuan_api_key).get_secret_value() == "test-secret-key"
    )
