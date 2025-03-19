from typing import Any

import pytest
from pydantic import SecretStr

from langchain_community.utilities.brave_search import BraveSearchWrapper


def test_api_key_explicit() -> None:
    """Test that the API key is correctly set when provided explicitly."""
    explicit_key = "explicit-api-key"
    wrapper = BraveSearchWrapper(api_key=SecretStr(explicit_key), search_kwargs={})
    assert wrapper.api_key.get_secret_value() == explicit_key


def test_api_key_from_env(monkeypatch: Any) -> None:
    """Test that the API key is correctly obtained from the environment variable."""
    env_key = "env-api-key"
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", env_key)
    # Do not pass the api_key explicitly
    wrapper = BraveSearchWrapper()  # type: ignore[call-arg]
    assert wrapper.api_key.get_secret_value() == env_key


def test_api_key_missing(monkeypatch: Any) -> None:
    """Test that instantiation fails when no API key is provided
    either explicitly or via environment."""
    # Ensure that the environment variable is not set
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    with pytest.raises(ValueError):
        # This should raise an error because no api_key is available.
        BraveSearchWrapper()  # type: ignore[call-arg]
