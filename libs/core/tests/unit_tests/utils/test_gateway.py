"""Tests for LangSmith Gateway URL configuration."""

from typing import get_type_hints

import pytest

from langchain_core.utils.gateway import (
    LANGSMITH_GATEWAY_PROVIDERS,
    LANGSMITH_GATEWAY_URL,
    GatewayProviderConfig,
    resolve_langsmith_gateway_url,
)


@pytest.mark.parametrize(
    ("provider", "expected_url"),
    [
        ("anthropic", f"{LANGSMITH_GATEWAY_URL}/anthropic"),
        ("fireworks", f"{LANGSMITH_GATEWAY_URL}/fireworks"),
        ("openai", f"{LANGSMITH_GATEWAY_URL}/openai/v1"),
    ],
)
def test_resolve_langsmith_gateway_url_for_hosted_gateway(
    monkeypatch: pytest.MonkeyPatch, provider: str, expected_url: str
) -> None:
    """A true value selects the hosted URL for every supported provider."""
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")

    assert resolve_langsmith_gateway_url(provider) == expected_url  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("provider", "expected_url"),
    [
        ("anthropic", "https://gateway.example.com/anthropic"),
        ("fireworks", "https://gateway.example.com/fireworks"),
        ("openai", "https://gateway.example.com/openai/v1"),
    ],
)
def test_resolve_langsmith_gateway_url_for_custom_root(
    monkeypatch: pytest.MonkeyPatch, provider: str, expected_url: str
) -> None:
    """A custom root gets the provider's centrally defined path."""
    monkeypatch.setenv("LANGSMITH_GATEWAY", "https://gateway.example.com/")

    assert resolve_langsmith_gateway_url(provider) == expected_url  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("provider", "url"),
    [
        ("anthropic", "https://gateway.example.com/anthropic/"),
        ("fireworks", "https://gateway.example.com/fireworks"),
        ("openai", "https://gateway.example.com/openai/v1"),
    ],
)
def test_resolve_langsmith_gateway_url_preserves_provider_url(
    monkeypatch: pytest.MonkeyPatch, provider: str, url: str
) -> None:
    """A provider-qualified custom URL is not modified."""
    monkeypatch.setenv("LANGSMITH_GATEWAY", url)

    assert resolve_langsmith_gateway_url(provider) == url  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["false", "0", "no"])
def test_resolve_langsmith_gateway_url_disabled(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Disabled Gateway values leave provider configuration unchanged."""
    monkeypatch.setenv("LANGSMITH_GATEWAY", value)

    assert resolve_langsmith_gateway_url("openai") is None


def test_resolve_langsmith_gateway_url_rejects_unsupported_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The published matrix makes Google GenAI's unsupported status explicit."""
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")

    with pytest.raises(ValueError, match="does not support provider `google_genai`"):
        resolve_langsmith_gateway_url("google_genai")


def test_langsmith_gateway_provider_matrix() -> None:
    """The initial support matrix is machine-consumable."""
    assert LANGSMITH_GATEWAY_PROVIDERS == {
        "anthropic": {"path": "/anthropic", "oauth": True, "static_key": True},
        "fireworks": {"path": "/fireworks", "oauth": True, "static_key": True},
        "google_genai": {
            "path": "/google-genai",
            "oauth": False,
            "static_key": False,
            "notes": "Not supported by LangSmith Gateway.",
        },
        "openai": {"path": "/openai/v1", "oauth": True, "static_key": True},
    }


@pytest.mark.parametrize(
    ("provider", "url", "expected_url"),
    [
        (
            "openai",
            "https://gateway.example.com?api-version=1#gateway",
            "https://gateway.example.com/openai/v1?api-version=1#gateway",
        ),
        (
            "openai",
            "https://gateway.example.com/openai/v1?api-version=1#gateway",
            "https://gateway.example.com/openai/v1?api-version=1#gateway",
        ),
    ],
)
def test_resolve_langsmith_gateway_url_preserves_query_and_fragment(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    url: str,
    expected_url: str,
) -> None:
    """Provider paths are added before URL query and fragment components."""
    monkeypatch.setenv("LANGSMITH_GATEWAY", url)

    assert resolve_langsmith_gateway_url(provider) == expected_url  # type: ignore[arg-type]


def test_gateway_provider_config_runtime_type_metadata() -> None:
    """The public provider configuration retains optional-key metadata at runtime."""
    assert GatewayProviderConfig.__optional_keys__ == {"notes"}
    assert get_type_hints(GatewayProviderConfig)["notes"] is str
