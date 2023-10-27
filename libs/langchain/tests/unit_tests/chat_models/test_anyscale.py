"""Test Anyscale chat model."""
from typing import List

import pytest
from pytest import CaptureFixture, MonkeyPatch

from langchain.chat_models.anyscale import DEFAULT_MODEL, ChatAnyscale
from langchain.pydantic_v1 import SecretStr


@pytest.fixture(autouse=True)
def prevent_anyscale_api_request(monkeypatch: MonkeyPatch) -> None:
    def mocked_get_available_models(arg1: None, arg2: None) -> List[str]:
        return [DEFAULT_MODEL]

    monkeypatch.setattr(
        ChatAnyscale,
        "get_available_models",
        mocked_get_available_models,
    )


@pytest.mark.requires("openai")
def test_api_key_is_secret_string() -> None:
    llm = ChatAnyscale(
        anyscale_api_key="secret-api-key", openai_api_key="secret-api-key"
    )
    assert isinstance(llm.anyscale_api_key, SecretStr)


@pytest.mark.requires("openai")
def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    monkeypatch.setenv("ANYSCALE_API_KEY", "secret-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-api-key")
    llm = ChatAnyscale()
    print(llm.anyscale_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


@pytest.mark.requires("openai")
def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = ChatAnyscale(
        anyscale_api_key="secret-api-key", openai_api_key="secret-api-key"
    )
    print(llm.anyscale_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
