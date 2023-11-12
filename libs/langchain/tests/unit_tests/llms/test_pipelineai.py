"""Test Mystic AI's PipelineAI module."""

from pytest import CaptureFixture, MonkeyPatch

from langchain.llms.pipelineai import PipelineAI
from langchain.pydantic_v1 import SecretStr


def test_api_key_is_secret_string() -> None:
    llm = PipelineAI(pipeline_api_key="secret-api-key")
    assert isinstance(llm.pipeline_api_key, SecretStr)


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("PIPELINE_API_KEY", "secret-api-key")
    llm = PipelineAI()
    print(llm.pipeline_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
    assert str(llm.pipeline_api_key) == "**********"
    assert "secret-api-key" not in repr(llm.pipeline_api_key)
    assert "secret-api-key" not in repr(llm)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    llm = PipelineAI(pipeline_api_key="secret-api-key")
    print(llm.pipeline_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
    assert str(llm.pipeline_api_key) == "**********"
    assert "secret-api-key" not in repr(llm.pipeline_api_key)
    assert "secret-api-key" not in repr(llm)
