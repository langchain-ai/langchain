"""Test Google PaLM Text API wrapper.

Note: This test must be run with the GOOGLE_API_KEY environment variable set to a
      valid API key.
"""

from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

from langchain.llms.google_palm import GooglePalm
from langchain.llms.loading import load_llm
from langchain.pydantic_v1 import SecretStr


def test_api_key_is_secret_string() -> None:
    llm = GooglePalm(google_api_key="secret_api_key")
    assert isinstance(llm.google_api_key, SecretStr)


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-api-key")
    llm = GooglePalm(google_api_key="secret_api_key")
    print(llm.google_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
    assert str(llm.google_api_key) == "**********"
    assert "secret-api-key" not in repr(llm.google_api_key)
    assert "secret-api-key" not in repr(llm)


def test_api_key_masked_when_passed_via_constructor(capsys: CaptureFixture) -> None:
    """Test initialization with an API key provided via the constructor"""
    llm = GooglePalm(google_api_key="secret-api-key")
    print(llm.google_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
    assert str(llm.google_api_key) == "**********"
    assert "secret-api-key" not in repr(llm.google_api_key)
    assert "secret-api-key" not in repr(llm)


def test_google_palm_call() -> None:
    """Test valid call to Google PaLM text API."""
    llm = GooglePalm(max_output_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading a Google PaLM LLM."""
    llm = GooglePalm(max_output_tokens=10)
    llm.save(file_path=tmp_path / "google_palm.yaml")
    loaded_llm = load_llm(tmp_path / "google_palm.yaml")
    assert loaded_llm == llm
