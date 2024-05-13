"""Test JinaChat wrapper."""

from typing import cast

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.chat_models.jinachat import JinaChat
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_jinachat_api_key_is_secret_string() -> None:
    llm = JinaChat(jinachat_api_key="secret-api-key")  # type: ignore[arg-type, call-arg]
    assert isinstance(llm.jinachat_api_key, SecretStr)


def test_jinachat_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("JINACHAT_API_KEY", "secret-api-key")
    llm = JinaChat()  # type: ignore[call-arg]
    print(llm.jinachat_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_jinachat_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    llm = JinaChat(jinachat_api_key="secret-api-key")  # type: ignore[arg-type, call-arg]
    print(llm.jinachat_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secretstr() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    llm = JinaChat(jinachat_api_key="secret-api-key")  # type: ignore[arg-type, call-arg]
    assert cast(SecretStr, llm.jinachat_api_key).get_secret_value() == "secret-api-key"


def test_jinachat() -> None:
    """Test JinaChat wrapper."""
    chat = JinaChat(max_tokens=10)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_jinachat_system_message() -> None:
    """Test JinaChat wrapper with system message."""
    chat = JinaChat(max_tokens=10)  # type: ignore[call-arg]
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_jinachat_generate() -> None:
    """Test JinaChat wrapper with generate."""
    chat = JinaChat(max_tokens=10)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test_jinachat_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = JinaChat(  # type: ignore[call-arg]
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


async def test_async_jinachat() -> None:
    """Test async generation."""
    chat = JinaChat(max_tokens=102)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


async def test_async_jinachat_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = JinaChat(  # type: ignore[call-arg]
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test_jinachat_extra_kwargs() -> None:
    """Test extra kwargs to chat openai."""
    # Check that foo is saved in extra_kwargs.
    llm = JinaChat(foo=3, max_tokens=10)  # type: ignore[call-arg]
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = JinaChat(foo=3, model_kwargs={"bar": 2})  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        JinaChat(foo=3, model_kwargs={"foo": 2})  # type: ignore[call-arg]

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        JinaChat(model_kwargs={"temperature": 0.2})  # type: ignore[call-arg]
