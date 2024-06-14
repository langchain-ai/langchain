"""Test ForeFrontAI LLM"""

from typing import cast

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.llms.forefrontai import ForefrontAI


def test_forefrontai_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    llm = ForefrontAI(forefrontai_api_key="secret-api-key", temperature=0.2)  # type: ignore[arg-type]
    assert isinstance(llm.forefrontai_api_key, SecretStr)


def test_forefrontai_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("FOREFRONTAI_API_KEY", "secret-api-key")
    llm = ForefrontAI(temperature=0.2)  # type: ignore[call-arg]
    print(llm.forefrontai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_forefrontai_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    llm = ForefrontAI(
        forefrontai_api_key="secret-api-key",  # type: ignore[arg-type]
        temperature=0.2,
    )
    print(llm.forefrontai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_forefrontai_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    llm = ForefrontAI(
        forefrontai_api_key="secret-api-key",  # type: ignore[arg-type]
        temperature=0.2,
    )
    assert (
        cast(SecretStr, llm.forefrontai_api_key).get_secret_value() == "secret-api-key"
    )
