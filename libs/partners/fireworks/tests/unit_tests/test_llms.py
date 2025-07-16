"""Test Fireworks LLM."""

from typing import cast

from pydantic import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_fireworks import Fireworks


def test_fireworks_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    llm = Fireworks(  # type: ignore[call-arg]
        fireworks_api_key="secret-api-key",
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.2,
        max_tokens=250,
    )
    assert isinstance(llm.fireworks_api_key, SecretStr)

    # Test api_key alias
    llm = Fireworks(
        api_key="secret-api-key",  # type: ignore[arg-type]
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.2,
        max_tokens=250,
    )
    assert isinstance(llm.fireworks_api_key, SecretStr)


def test_fireworks_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "secret-api-key")
    llm = Fireworks(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.2,
        max_tokens=250,
    )
    print(llm.fireworks_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_fireworks_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    llm = Fireworks(  # type: ignore[call-arg]
        fireworks_api_key="secret-api-key",
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.2,
        max_tokens=250,
    )
    print(llm.fireworks_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_fireworks_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    llm = Fireworks(  # type: ignore[call-arg]
        fireworks_api_key="secret-api-key",
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.2,
        max_tokens=250,
    )
    assert cast(SecretStr, llm.fireworks_api_key).get_secret_value() == "secret-api-key"


def test_fireworks_model_params() -> None:
    # Test standard tracing params
    llm = Fireworks(model="foo", api_key="secret-api-key")  # type: ignore[arg-type]

    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "fireworks",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
    }

    llm = Fireworks(
        model="foo",
        api_key="secret-api-key",  # type: ignore[arg-type]
        max_tokens=10,
        temperature=0.1,
    )

    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "fireworks",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_max_tokens": 10,
        "ls_temperature": 0.1,
    }
