"""Test client utility functions."""

from __future__ import annotations

from langchain_anthropic._client_utils import (
    _get_default_async_httpx_client,
    _get_default_httpx_client,
)


def test_sync_client_without_proxy() -> None:
    """Test sync client creation without proxy."""
    client = _get_default_httpx_client(base_url="https://api.anthropic.com")

    # Should not have proxy configured
    assert not hasattr(client, "proxies") or client.proxies is None


def test_sync_client_with_proxy() -> None:
    """Test sync client creation with proxy."""
    proxy_url = "http://proxy.example.com:8080"
    client = _get_default_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=proxy_url
    )

    # Check internal _transport since httpx stores proxy configuration in the transport
    # layer
    transport = getattr(client, "_transport", None)
    assert transport is not None


def test_async_client_without_proxy() -> None:
    """Test async client creation without proxy."""
    client = _get_default_async_httpx_client(base_url="https://api.anthropic.com")

    assert not hasattr(client, "proxies") or client.proxies is None


def test_async_client_with_proxy() -> None:
    """Test async client creation with proxy."""
    proxy_url = "http://proxy.example.com:8080"
    client = _get_default_async_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=proxy_url
    )

    transport = getattr(client, "_transport", None)
    assert transport is not None


def test_client_proxy_none_value() -> None:
    """Test that explicitly passing None for proxy works correctly."""
    sync_client = _get_default_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=None
    )

    async_client = _get_default_async_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=None
    )

    # Both should be created successfully with None proxy
    assert sync_client is not None
    assert async_client is not None
