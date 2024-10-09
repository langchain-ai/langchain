"""Test Aleph Alpha specific stuff."""

import pytest
from pydantic import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.llms.aleph_alpha import AlephAlpha


@pytest.mark.requires("aleph_alpha_client")
def test_api_key_is_secret_string() -> None:
    llm = AlephAlpha(aleph_alpha_api_key="secret-api-key")  # type: ignore
    assert isinstance(llm.aleph_alpha_api_key, SecretStr)


@pytest.mark.requires("aleph_alpha_client")
def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = AlephAlpha(aleph_alpha_api_key="secret-api-key")  # type: ignore
    print(llm.aleph_alpha_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


@pytest.mark.requires("aleph_alpha_client")
def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    monkeypatch.setenv("ALEPH_ALPHA_API_KEY", "secret-api-key")
    llm = AlephAlpha()  # type: ignore[call-arg]
    print(llm.aleph_alpha_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"
