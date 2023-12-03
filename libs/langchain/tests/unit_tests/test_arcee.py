"""Test Arcee llm"""
import pytest
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain.llms.arcee import Arcee


@pytest.mark.requires("arcee")
def test_api_key_is_secret_string() -> None:
    llm = Arcee(arcee_api_key="test-arcee-api-key")
    assert isinstance(llm.arcee_api_key, SecretStr)


@pytest.mark.requires("arcee")
def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("ARCEE-API-KEY", "test-arcee-api-key")

    llm = Arcee()

    print(llm.arcee_api_key, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"


@pytest.mark.requires("arcee")
def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    llm = Arcee(arcee_api_key="test-arcee-api-key")

    print(llm.arcee_api_key, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"
