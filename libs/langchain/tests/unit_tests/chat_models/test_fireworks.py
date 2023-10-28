"""Test Fireworks chat model"""

import pytest
from pytest import CaptureFixture, MonkeyPatch

from langchain.chat_models import ChatFireworks
from langchain.pydantic_v1 import SecretStr


@pytest.mark.requires("openai")
def test_api_key_is_string() -> None:
    llm = ChatFireworks(fireworks_api_key="secret-api-key")
    assert isinstance(llm.fireworks_api_key, SecretStr)


@pytest.mark.requires("openai")
def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "secret-api-key")
    llm = ChatFireworks()
    print(llm.fireworks_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


@pytest.mark.requires("openai")
def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = ChatFireworks(fireworks_api_key="secret-api-key")
    print(llm.fireworks_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
