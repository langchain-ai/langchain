"""Test Aleph Alpha specific stuff."""

import pytest
from pytest import CaptureFixture, MonkeyPatch

from langchain.llms.aleph_alpha import AlephAlpha
from langchain.pydantic_v1 import SecretStr


@pytest.mark.requires("aleph_alpha_client")
def test_api_key_is_secret_string() -> None:
    llm = AlephAlpha(aleph_alpha_api_key="secret-api-key")
    assert isinstance(llm.aleph_alpha_api_key, SecretStr)


@pytest.mark.requires("aleph_alpha_client")
def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = AlephAlpha(aleph_alpha_api_key="secret-api-key")
    print(llm.aleph_alpha_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


@pytest.mark.requires("aleph_alpha_client")
def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    monkeypatch.setenv("ALEPH_ALPHA_API_KEY", "secret-api-key")
    llm = AlephAlpha()
    print(llm.aleph_alpha_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
