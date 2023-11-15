"""Test Anyscale llm"""
import pytest
from pytest import CaptureFixture, MonkeyPatch

from langchain.llms.anyscale import Anyscale
from langchain.pydantic_v1 import SecretStr


@pytest.mark.requires("openai")
def test_api_key_is_secret_string() -> None:
    llm = Anyscale(
        anyscale_api_key="secret-api-key", anyscale_api_base="test", model_name="test"
    )
    assert isinstance(llm.anyscale_api_key, SecretStr)


@pytest.mark.requires("openai")
def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("ANYSCALE_API_KEY", "secret-api-key")
    llm = Anyscale(anyscale_api_base="test", model_name="test")
    print(llm.anyscale_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


@pytest.mark.requires("openai")
def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    llm = Anyscale(
        anyscale_api_key="secret-api-key", anyscale_api_base="test", model_name="test"
    )
    print(llm.anyscale_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
