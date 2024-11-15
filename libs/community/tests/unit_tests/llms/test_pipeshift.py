"""Test Pipeshift Chat API wrapper."""

from typing import cast

from pydantic import SecretStr  # type: ignore[import-not-found]
from pytest import CaptureFixture, MonkeyPatch  # type: ignore[import-not-found]

from langchain_community.llms.pipeshift import Pipeshift


def test_pipeshift_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    llm = Pipeshift(
        pipeshift_api_key="secret-api-key",  # type: ignore[arg-type]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=512,
    )
    assert isinstance(llm.pipeshift_api_key, SecretStr)


def test_pipeshift_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("PIPESHIFT_API_KEY", "secret-api-key")
    llm = Pipeshift(  # type: ignore[call-arg]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=512,
    )
    print(llm.pipeshift_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_pipeshift_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    llm = Pipeshift(
        pipeshift_api_key="secret-api-key",  # type: ignore[arg-type]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=512,
    )
    print(llm.pipeshift_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_pipeshift_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    llm = Pipeshift(
        pipeshift_api_key="secret-api-key",  # type: ignore[arg-type]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=512,
    )
    assert cast(SecretStr, llm.pipeshift_api_key).get_secret_value() == "secret-api-key"


def test_pipeshift_uses_actual_secret_value_from_secretstr_api_key() -> None:
    """Test that the actual secret value is correctly retrieved."""
    llm = Pipeshift(
        pipeshift_api_key="secret-api-key",  # type: ignore[arg-type]
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.2,
        max_tokens=512,
    )
    assert cast(SecretStr, llm.pipeshift_api_key).get_secret_value() == "secret-api-key"


def test_pipeshift_model_params() -> None:
    # Test standard tracing params
    llm = Pipeshift(
        pipeshift_api_key="secret-api-key",  # type: ignore[arg-type]
        model="foo",
        max_tokens=512,
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "pipeshift",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_max_tokens": 512,
    }

    llm = Pipeshift(
        pipeshift_api_key="secret-api-key",  # type: ignore[arg-type]
        model="foo",
        temperature=0.2,
        max_tokens=512,
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "pipeshift",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_max_tokens": 512,
        "ls_temperature": 0.2,
    }
