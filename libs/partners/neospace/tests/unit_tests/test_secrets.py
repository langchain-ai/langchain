from typing import Type, cast

import pytest
from langchain_core.load import dumpd
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_neospace import (
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
)


def test_chat_openai_secrets() -> None:
    o = ChatOpenAI(openai_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s


def test_openai_secrets() -> None:
    o = OpenAI(openai_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s


def test_openai_embeddings_secrets() -> None:
    o = OpenAIEmbeddings(openai_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s


@pytest.mark.parametrize("model_class", [ChatOpenAI, OpenAI, OpenAIEmbeddings])
def test_openai_api_key_is_secret_string(model_class: Type) -> None:
    """Test that the API key is stored as a SecretStr."""
    model = model_class(openai_api_key="secret-api-key")
    assert isinstance(model.openai_api_key, SecretStr)


@pytest.mark.parametrize("model_class", [ChatOpenAI, OpenAI, OpenAIEmbeddings])
def test_openai_api_key_masked_when_passed_from_env(
    model_class: Type, monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret-api-key")
    model = model_class()
    print(model.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


@pytest.mark.parametrize("model_class", [ChatOpenAI, OpenAI, OpenAIEmbeddings])
def test_openai_api_key_masked_when_passed_via_constructor(
    model_class: Type, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    model = model_class(openai_api_key="secret-api-key")
    print(model.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


@pytest.mark.parametrize("model_class", [ChatOpenAI, OpenAI, OpenAIEmbeddings])
def test_openai_uses_actual_secret_value_from_secretstr(model_class: Type) -> None:
    """Test that the actual secret value is correctly retrieved."""
    model = model_class(openai_api_key="secret-api-key")
    assert cast(SecretStr, model.openai_api_key).get_secret_value() == "secret-api-key"
