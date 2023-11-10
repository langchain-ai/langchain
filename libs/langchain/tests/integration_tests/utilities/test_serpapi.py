"""Integration test for SerpApi."""
from typing import cast

import pytest
from pytest import CaptureFixture, MonkeyPatch

from langchain.pydantic_v1 import SecretStr
from langchain.utilities import SerpAPIWrapper


def test_google_call() -> None:
    """Test that call gives the correct answer."""
    chain = SerpAPIWrapper()
    output = chain.run("What was Obama's first name?")
    assert output == "Barack Hussein Obama II"


def test_bing_call() -> None:
    """Test that call gives the correct answer."""
    chain = SerpAPIWrapper(params={"engine": "bing"})
    output = chain.run("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output


def test_api_key_is_secret_string() -> None:
    chain = SerpAPIWrapper(serpapi_api_key="secret-api-key")
    assert isinstance(chain.serpapi_api_key, SecretStr)


def test_api_key_is_secret_none(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("SERPAPI_API_KEY")

    with pytest.raises(ValueError):
        chain = SerpAPIWrapper()
        assert chain.serpapi_api_key is None


def test_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("SERPAPI_API_KEY", "secret-api-key")
    chain = SerpAPIWrapper()
    print(chain.serpapi_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
    assert str(chain.serpapi_api_key) == "**********"
    assert "secret-api-key" not in repr(chain.serpapi_api_key)
    assert "secret-api-key" not in repr(chain)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    chain = SerpAPIWrapper(serpapi_api_key="secret-api-key")
    print(chain.serpapi_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
    assert str(chain.serpapi_api_key) == "**********"
    assert "secret-api-key" not in repr(chain.serpapi_api_key)
    assert "secret-api-key" not in repr(chain)


def test_uses_actual_secret_value_from_secretstr() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    chain = SerpAPIWrapper(serpapi_api_key="secret-api-key")
    assert cast(SecretStr, chain.serpapi_api_key).get_secret_value() == "secret-api-key"
