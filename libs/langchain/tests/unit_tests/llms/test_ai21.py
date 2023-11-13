"""Test AI21 llm"""
from typing import cast

from pytest import CaptureFixture, MonkeyPatch

from langchain.llms.ai21 import AI21
from langchain.pydantic_v1 import SecretStr


def test_api_key_is_secret_string() -> None:
    llm = AI21(ai21_api_key="secret-api-key")
    assert isinstance(llm.ai21_api_key, SecretStr)


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("AI21_API_KEY", "secret-api-key")
    llm = AI21()
    print(llm.ai21_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    llm = AI21(ai21_api_key="secret-api-key")
    print(llm.ai21_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secretstr() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    llm = AI21(ai21_api_key="secret-api-key")
    assert cast(SecretStr, llm.ai21_api_key).get_secret_value() == "secret-api-key"
