"""Test Arcee llm"""
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain.llms.arcee import Arcee


def test_api_key_is_secret_string() -> None:
    llm = Arcee(model="DALM-PubMed", arcee_api_key="test-arcee-api-key")
    assert isinstance(llm.arcee_api_key, SecretStr)


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("ARCEE_API_KEY", "test-arcee-api-key")

    llm = Arcee(model="DALM-PubMed")

    print(llm.arcee_api_key, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    llm = Arcee(model="DALM-PubMed", arcee_api_key="test-arcee-api-key")

    print(llm.arcee_api_key, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"
