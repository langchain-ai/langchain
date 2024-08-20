"""Evaluate ChatKonko Interface."""

from typing import Any, cast

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.chat_models.konko import ChatKonko
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_konko_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("KONKO_API_KEY", "test-konko-key")

    chat = ChatKonko()

    print(chat.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"

    print(chat.konko_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"


def test_konko_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    chat = ChatKonko(openai_api_key="test-openai-key", konko_api_key="test-konko-key")

    print(chat.konko_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"

    print(chat.konko_secret_key, end="")  # type: ignore[attr-defined] # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secret_str() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    chat = ChatKonko(openai_api_key="test-openai-key", konko_api_key="test-konko-key")
    assert cast(SecretStr, chat.konko_api_key).get_secret_value() == "test-openai-key"
    assert cast(SecretStr, chat.konko_secret_key).get_secret_value() == "test-konko-key"  # type: ignore[attr-defined]


def test_konko_chat_test() -> None:
    """Evaluate basic ChatKonko functionality."""
    chat_instance = ChatKonko(max_tokens=10)
    msg = HumanMessage(content="Hi")
    chat_response = chat_instance.invoke([msg])
    assert isinstance(chat_response, BaseMessage)
    assert isinstance(chat_response.content, str)


def test_konko_chat_test_openai() -> None:
    """Evaluate basic ChatKonko functionality."""
    chat_instance = ChatKonko(max_tokens=10, model="meta-llama/llama-2-70b-chat")
    msg = HumanMessage(content="Hi")
    chat_response = chat_instance.invoke([msg])
    assert isinstance(chat_response, BaseMessage)
    assert isinstance(chat_response.content, str)


def test_konko_model_test() -> None:
    """Check how ChatKonko manages model_name."""
    chat_instance = ChatKonko(model="alpha")
    assert chat_instance.model == "alpha"
    chat_instance = ChatKonko(model="beta")
    assert chat_instance.model == "beta"


def test_konko_available_model_test() -> None:
    """Check how ChatKonko manages model_name."""
    chat_instance = ChatKonko(max_tokens=10, n=2)
    res = chat_instance.get_available_models()
    assert isinstance(res, set)


def test_konko_system_msg_test() -> None:
    """Evaluate ChatKonko's handling of system messages."""
    chat_instance = ChatKonko(max_tokens=10)
    sys_msg = SystemMessage(content="Initiate user chat.")
    user_msg = HumanMessage(content="Hi there")
    chat_response = chat_instance.invoke([sys_msg, user_msg])
    assert isinstance(chat_response, BaseMessage)
    assert isinstance(chat_response.content, str)


def test_konko_generation_test() -> None:
    """Check ChatKonko's generation ability."""
    chat_instance = ChatKonko(max_tokens=10, n=2)
    msg = HumanMessage(content="Hi")
    gen_response = chat_instance.generate([[msg], [msg]])
    assert isinstance(gen_response, LLMResult)
    assert len(gen_response.generations) == 2
    for gen_list in gen_response.generations:
        assert len(gen_list) == 2
        for gen in gen_list:
            assert isinstance(gen, ChatGeneration)
            assert isinstance(gen.text, str)
            assert gen.text == gen.message.content


def test_konko_multiple_outputs_test() -> None:
    """Test multiple completions with ChatKonko."""
    chat_instance = ChatKonko(max_tokens=10, n=5)
    msg = HumanMessage(content="Hi")
    gen_response = chat_instance._generate([msg])
    assert isinstance(gen_response, ChatResult)
    assert len(gen_response.generations) == 5
    for gen in gen_response.generations:
        assert isinstance(gen.message, BaseMessage)
        assert isinstance(gen.message.content, str)


def test_konko_streaming_callback_test() -> None:
    """Evaluate streaming's token callback functionality."""
    callback_instance = FakeCallbackHandler()
    callback_mgr = CallbackManager([callback_instance])
    chat_instance = ChatKonko(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_mgr,
        verbose=True,
    )
    msg = HumanMessage(content="Hi")
    chat_response = chat_instance.invoke([msg])
    assert callback_instance.llm_streams > 0
    assert isinstance(chat_response, BaseMessage)


def test_konko_streaming_info_test() -> None:
    """Ensure generation details are retained during streaming."""

    class TestCallback(FakeCallbackHandler):
        data_store: dict = {}

        def on_llm_end(self, *args: Any, **kwargs: Any) -> Any:
            self.data_store["generation"] = args[0]

    callback_instance = TestCallback()
    callback_mgr = CallbackManager([callback_instance])
    chat_instance = ChatKonko(
        max_tokens=2,
        temperature=0,
        callback_manager=callback_mgr,
    )
    list(chat_instance.stream("hey"))
    gen_data = callback_instance.data_store["generation"]
    assert gen_data.generations[0][0].text == " Hey"


def test_konko_llm_model_name_test() -> None:
    """Check if llm_output has model info."""
    chat_instance = ChatKonko(max_tokens=10)
    msg = HumanMessage(content="Hi")
    llm_data = chat_instance.generate([[msg]])
    assert llm_data.llm_output is not None
    assert llm_data.llm_output["model_name"] == chat_instance.model


def test_konko_streaming_model_name_test() -> None:
    """Check model info during streaming."""
    chat_instance = ChatKonko(max_tokens=10, streaming=True)
    msg = HumanMessage(content="Hi")
    llm_data = chat_instance.generate([[msg]])
    assert llm_data.llm_output is not None
    assert llm_data.llm_output["model_name"] == chat_instance.model


def test_konko_streaming_param_validation_test() -> None:
    """Ensure correct token callback during streaming."""
    with pytest.raises(ValueError):
        ChatKonko(
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )


def test_konko_additional_args_test() -> None:
    """Evaluate extra arguments for ChatKonko."""
    chat_instance = ChatKonko(extra=3, max_tokens=10)  # type: ignore[call-arg]
    assert chat_instance.max_tokens == 10
    assert chat_instance.model_kwargs == {"extra": 3}

    chat_instance = ChatKonko(extra=3, model_kwargs={"addition": 2})  # type: ignore[call-arg]
    assert chat_instance.model_kwargs == {"extra": 3, "addition": 2}

    with pytest.raises(ValueError):
        ChatKonko(extra=3, model_kwargs={"extra": 2})  # type: ignore[call-arg]

    with pytest.raises(ValueError):
        ChatKonko(model_kwargs={"temperature": 0.2})

    with pytest.raises(ValueError):
        ChatKonko(model_kwargs={"model": "gpt-3.5-turbo-instruct"})


def test_konko_token_streaming_test() -> None:
    """Check token streaming for ChatKonko."""
    chat_instance = ChatKonko(max_tokens=10)

    for token in chat_instance.stream("Just a test"):
        assert isinstance(token.content, str)
