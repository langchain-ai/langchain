"""Test Minimax llm"""

from typing import cast

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.llms.minimax import Minimax


def test_api_key_is_secret_string() -> None:
    llm = Minimax(minimax_api_key="secret-api-key", minimax_group_id="group_id")  # type: ignore[arg-type, call-arg]
    assert isinstance(llm.minimax_api_key, SecretStr)


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("MINIMAX_API_KEY", "secret-api-key")
    monkeypatch.setenv("MINIMAX_GROUP_ID", "group_id")
    llm = Minimax()  # type: ignore[call-arg]
    print(llm.minimax_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    llm = Minimax(minimax_api_key="secret-api-key", minimax_group_id="group_id")  # type: ignore[arg-type, call-arg]
    print(llm.minimax_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secretstr() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    llm = Minimax(minimax_api_key="secret-api-key", minimax_group_id="group_id")  # type: ignore[arg-type, call-arg]
    assert cast(SecretStr, llm.minimax_api_key).get_secret_value() == "secret-api-key"
