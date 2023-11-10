"""Integration test for SerpApi's Google Scholar API."""
from typing import cast

import pytest
from pytest import CaptureFixture, MonkeyPatch

from langchain.pydantic_v1 import SecretStr
from langchain.utilities import GoogleScholarAPIWrapper


def test_api_key_is_secret_string() -> None:
    chain = GoogleScholarAPIWrapper(serp_api_key="secret-api-key")
    assert isinstance(chain.serp_api_key, SecretStr)


def test_api_key_is_secret_none(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("SERPAPI_API_KEY")

    with pytest.raises(ValueError):
        chain = GoogleScholarAPIWrapper()
        assert chain.serpapi_api_key is None


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("SERPAPI_API_KEY", "secret-api-key")
    chain = GoogleScholarAPIWrapper()
    print(chain.serp_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
    assert str(chain.serp_api_key) == "**********"
    assert "secret-api-key" not in repr(chain.serp_api_key)
    assert "secret-api-key" not in repr(chain)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    chain = GoogleScholarAPIWrapper(serp_api_key="secret-api-key")
    print(chain.serp_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
    assert str(chain.serp_api_key) == "**********"
    assert "secret-api-key" not in repr(chain.serp_api_key)
    assert "secret-api-key" not in repr(chain)


def test_uses_actual_secret_value_from_secretstr() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    chain = GoogleScholarAPIWrapper(serp_api_key="secret-api-key")
    assert cast(SecretStr, chain.serp_api_key).get_secret_value() == "secret-api-key"
