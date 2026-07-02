"""Tests for Azure OpenAI environment variable helpers."""

from __future__ import annotations

import warnings

import pytest

from langchain_openai._azure_env import azure_openai_api_key_from_env


def test_azure_key_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """``AZURE_OPENAI_API_KEY`` must be preferred over ``OPENAI_API_KEY``."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    key = azure_openai_api_key_from_env()
    assert key is not None
    assert key.get_secret_value() == "azure-key"


def test_openai_key_fallback_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falling back to ``OPENAI_API_KEY`` should emit a deprecation warning."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        key = azure_openai_api_key_from_env()

    assert key is not None
    assert key.get_secret_value() == "openai-key"
    assert any(
        issubclass(w.category, DeprecationWarning) and "OPENAI_API_KEY" in str(w.message)
        for w in caught
    )


def test_no_key_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return ``None`` when neither env var is set."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert azure_openai_api_key_from_env() is None
