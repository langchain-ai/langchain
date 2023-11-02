from typing import cast

from pytest import CaptureFixture, MonkeyPatch

from langchain.llms import Petals
from langchain.pydantic_v1 import SecretStr


def test_api_key_is_secret_string() -> None:
    llm = Petals(huggingface_api_key="my-api-key")
    assert isinstance(llm.huggingface_api_key, SecretStr)


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "my-api-key")
    llm = Petals()
    print(llm.huggingface_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    llm = Petals(huggingface_api_key="my-api-key")
    print(llm.huggingface_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secretstr() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    llm = Petals(huggingface_api_key="my-api-key")
    assert cast(SecretStr, llm.huggingface_api_key).get_secret_value() == "my-api-key"