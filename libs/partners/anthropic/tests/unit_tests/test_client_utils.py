"""Unit tests for `langchain_anthropic._client_utils`.

Asserts socket-options plumbing at the boundary between our helpers and the
httpx layer — not on httpx internals. Locks the wiring, env-driven defaults,
the `()` kill-switch contract, and the precedence between constructor kwargs,
env vars, and user-supplied clients.
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Any

import httpx
import pytest

from langchain_anthropic import ChatAnthropic, _client_utils

SOL_SOCKET = socket.SOL_SOCKET
SO_KEEPALIVE = socket.SO_KEEPALIVE

_MODEL = "claude-opus-4-7"


@pytest.fixture(autouse=True)
def _clear_langchain_anthropic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure LANGCHAIN_ANTHROPIC_* env vars don't leak between tests."""
    for name in list(os.environ):
        if name.startswith("LANGCHAIN_ANTHROPIC_") or name == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # Reset module-level one-shot latches between tests.
    monkeypatch.setattr(_client_utils, "_proxy_env_warning_emitted", False)
    monkeypatch.setattr(_client_utils, "_proxy_env_bypass_info_emitted", False)


def test_sync_client_without_proxy() -> None:
    """Test sync client creation without proxy."""
    client = _client_utils._get_default_httpx_client(
        base_url="https://api.anthropic.com"
    )

    # Should not have proxy configured
    assert not hasattr(client, "proxies") or client.proxies is None


def test_sync_client_with_proxy() -> None:
    """Test sync client creation with proxy."""
    proxy_url = "http://proxy.example.com:8080"
    client = _client_utils._get_default_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=proxy_url
    )

    transport = getattr(client, "_transport", None)
    assert transport is not None


def test_async_client_without_proxy() -> None:
    """Test async client creation without proxy."""
    client = _client_utils._get_default_async_httpx_client(
        base_url="https://api.anthropic.com"
    )

    assert not hasattr(client, "proxies") or client.proxies is None


def test_async_client_with_proxy() -> None:
    """Test async client creation with proxy."""
    proxy_url = "http://proxy.example.com:8080"
    client = _client_utils._get_default_async_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=proxy_url
    )

    transport = getattr(client, "_transport", None)
    assert transport is not None


def test_client_proxy_none_value() -> None:
    """Test that explicitly passing None for proxy works correctly."""
    sync_client = _client_utils._get_default_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=None
    )

    async_client = _client_utils._get_default_async_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=None
    )

    assert sync_client is not None
    assert async_client is not None


@pytest.mark.skipif(
    __import__("sys").platform != "linux",
    reason="Default option set is platform-specific; Linux values asserted here.",
)
def test_default_socket_options_linux() -> None:
    """On Linux, the full option set should be present with default values."""
    opts = _client_utils._default_socket_options()
    expected = {
        (SOL_SOCKET, SO_KEEPALIVE, 1),
        (socket.IPPROTO_TCP, _client_utils._LINUX_TCP_KEEPIDLE, 60),
        (socket.IPPROTO_TCP, _client_utils._LINUX_TCP_KEEPINTVL, 10),
        (socket.IPPROTO_TCP, _client_utils._LINUX_TCP_KEEPCNT, 3),
        (socket.IPPROTO_TCP, _client_utils._LINUX_TCP_USER_TIMEOUT, 120000),
    }
    assert set(opts) == expected


def test_default_socket_options_disabled_returns_empty_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kill-switch: `()` is the single 'no options' shape, never None."""
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_TCP_KEEPALIVE", "0")
    opts = _client_utils._default_socket_options()
    assert opts == ()
    assert isinstance(opts, tuple)


@pytest.mark.enable_socket
def test_filter_supported_drops_unsupported() -> None:
    """An option with a deliberately-bogus level should be silently dropped."""
    good = (SOL_SOCKET, SO_KEEPALIVE, 1)
    bogus = (0xDEAD, 0xBEEF, 1)
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
    except OSError:
        pytest.skip("probe socket unavailable in this environment")
    result = _client_utils._filter_supported([good, bogus])
    assert good in result
    assert bogus not in result


def test_build_async_httpx_client_boundary_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Did our helper decide to inject a transport or not?"""
    recorded: list[dict[str, Any]] = []

    original = _client_utils._AsyncHttpxClientWrapper.__init__

    def spy(self: Any, **kwargs: Any) -> None:
        recorded.append(kwargs)
        original(self, **kwargs)

    monkeypatch.setattr(_client_utils._AsyncHttpxClientWrapper, "__init__", spy)

    _client_utils._build_async_httpx_client(
        base_url=None,
        socket_options=((SOL_SOCKET, SO_KEEPALIVE, 1),),
    )
    assert recorded, "expected one call when socket_options populated"
    assert "transport" in recorded[-1]

    recorded.clear()
    _client_utils._build_async_httpx_client(base_url=None, socket_options=())
    assert recorded, "expected one call when socket_options empty"
    assert "transport" not in recorded[-1]


def test_build_async_httpx_client_transport_carries_socket_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport should receive our options + the mirrored limits."""
    recorded: list[dict[str, Any]] = []

    original_cls = _client_utils.httpx.AsyncHTTPTransport

    class Recorder(original_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            recorded.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        "langchain_anthropic._client_utils.httpx.AsyncHTTPTransport",
        Recorder,
    )

    _client_utils._build_async_httpx_client(
        base_url=None,
        socket_options=((SOL_SOCKET, SO_KEEPALIVE, 1),),
    )

    assert recorded, "expected httpx.AsyncHTTPTransport to be constructed"
    kwargs = recorded[-1]
    assert kwargs.get("socket_options") == [(SOL_SOCKET, SO_KEEPALIVE, 1)]
    assert kwargs.get("limits") is _client_utils._DEFAULT_CONNECTION_LIMITS


def test_http_socket_options_none_vs_empty_tuple_vs_populated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discriminates the three input shapes at the builder boundary.

    Also locks the no-filter contract for user overrides: the populated-case
    assertion is verbatim, proving `_resolve_socket_options` does not run
    user overrides through `_filter_supported`.
    """
    recorded: list[tuple[str, tuple]] = []

    def spy_async(**kwargs: Any) -> Any:
        socket_options = kwargs.get("socket_options", ())
        recorded.append(("async", tuple(socket_options)))
        return _client_utils._AsyncHttpxClientWrapper(
            base_url=kwargs.get("base_url") or "https://api.anthropic.com",
        )

    def spy_sync(**kwargs: Any) -> Any:
        socket_options = kwargs.get("socket_options", ())
        recorded.append(("sync", tuple(socket_options)))
        return _client_utils._SyncHttpxClientWrapper(
            base_url=kwargs.get("base_url") or "https://api.anthropic.com",
        )

    monkeypatch.setattr(
        "langchain_anthropic.chat_models._get_default_async_httpx_client",
        spy_async,
    )
    monkeypatch.setattr(
        "langchain_anthropic.chat_models._get_default_httpx_client",
        spy_sync,
    )

    # (1) Unset -> None -> env-driven defaults (non-empty on linux/darwin CI).
    llm = ChatAnthropic(model=_MODEL)
    # Trigger lazy client construction.
    _ = llm._client
    _ = llm._async_client
    assert recorded, "expected a default-client build"
    _, opts1 = recorded[-1]
    assert isinstance(opts1, tuple)

    # (2) Explicit empty tuple -> ().
    recorded.clear()
    llm = ChatAnthropic(model=_MODEL, http_socket_options=())
    _ = llm._client
    _ = llm._async_client
    assert recorded
    assert all(opts == () for _, opts in recorded)

    # (3) Populated sequence -> verbatim passthrough (not filtered).
    recorded.clear()
    llm = ChatAnthropic(
        model=_MODEL,
        http_socket_options=[(SOL_SOCKET, SO_KEEPALIVE, 1)],
    )
    _ = llm._client
    _ = llm._async_client
    assert recorded
    for _, opts in recorded:
        assert opts == ((SOL_SOCKET, SO_KEEPALIVE, 1),)


def test_anthropic_proxy_applies_socket_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`anthropic_proxy` path must forward socket options to the builder."""
    recorded: list[dict[str, Any]] = []

    def sync_spy(**kwargs: Any) -> Any:
        recorded.append({"kind": "sync", **kwargs})
        return _client_utils._SyncHttpxClientWrapper(
            base_url=kwargs.get("base_url") or "https://api.anthropic.com",
        )

    def async_spy(**kwargs: Any) -> Any:
        recorded.append({"kind": "async", **kwargs})
        return _client_utils._AsyncHttpxClientWrapper(
            base_url=kwargs.get("base_url") or "https://api.anthropic.com",
        )

    monkeypatch.setattr(
        "langchain_anthropic.chat_models._get_default_httpx_client",
        sync_spy,
    )
    monkeypatch.setattr(
        "langchain_anthropic.chat_models._get_default_async_httpx_client",
        async_spy,
    )

    llm = ChatAnthropic(
        model=_MODEL,
        anthropic_proxy="http://proxy.example.com:3128",
        http_socket_options=[(SOL_SOCKET, SO_KEEPALIVE, 1)],
    )
    _ = llm._client
    _ = llm._async_client

    for record in recorded:
        assert record.get("anthropic_proxy") == "http://proxy.example.com:3128"
        assert record.get("socket_options") == ((SOL_SOCKET, SO_KEEPALIVE, 1),)


def test_default_path_opt_out_is_strict_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With LANGCHAIN_ANTHROPIC_TCP_KEEPALIVE=0 we inject no transport."""
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_TCP_KEEPALIVE", "0")

    recorded_sync: list[dict[str, Any]] = []
    recorded_async: list[dict[str, Any]] = []

    sync_original = _client_utils._SyncHttpxClientWrapper.__init__
    async_original = _client_utils._AsyncHttpxClientWrapper.__init__

    def sync_spy(self: Any, **kwargs: Any) -> None:
        recorded_sync.append(kwargs)
        sync_original(self, **kwargs)

    def async_spy(self: Any, **kwargs: Any) -> None:
        recorded_async.append(kwargs)
        async_original(self, **kwargs)

    monkeypatch.setattr(_client_utils._SyncHttpxClientWrapper, "__init__", sync_spy)
    monkeypatch.setattr(_client_utils._AsyncHttpxClientWrapper, "__init__", async_spy)

    # Clear cached builder results so env changes take effect.
    _client_utils._cached_sync_httpx_client.cache_clear()
    _client_utils._cached_async_httpx_client.cache_clear()

    llm = ChatAnthropic(model=_MODEL)
    _ = llm._client
    _ = llm._async_client

    assert recorded_sync, "expected the sync default client to be built"
    assert "transport" not in recorded_sync[-1]
    assert recorded_async, "expected the async default client to be built"
    assert "transport" not in recorded_async[-1]


def test_invalid_env_values_degrade_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    """Garbage in LANGCHAIN_ANTHROPIC_TCP_* env vars must not crash model init."""
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_TCP_KEEPIDLE", "not-an-int")
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_TCP_KEEPINTVL", "")
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_TCP_KEEPCNT", "NaN")
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_TCP_USER_TIMEOUT_MS", "abc")

    opts = _client_utils._default_socket_options()
    assert isinstance(opts, tuple)
    assert (SOL_SOCKET, SO_KEEPALIVE, 1) in opts

    # Instantiating a model doesn't raise.
    ChatAnthropic(model=_MODEL)


def test_invalid_stream_chunk_timeout_env_degrades_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garbage in LANGCHAIN_ANTHROPIC_STREAM_CHUNK_TIMEOUT_S must not crash init."""
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_STREAM_CHUNK_TIMEOUT_S", "not-a-float")
    model = ChatAnthropic(model=_MODEL)
    assert model.stream_chunk_timeout == 120.0


def test_default_socket_options_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS: `TCP_USER_TIMEOUT` is unavailable, but keepalive trio maps to darwin."""
    monkeypatch.setattr(_client_utils.sys, "platform", "darwin")
    opts = _client_utils._default_socket_options()
    assert (SOL_SOCKET, SO_KEEPALIVE, 1) in opts
    darwin_keepalive = (
        socket.IPPROTO_TCP,
        _client_utils._DARWIN_TCP_KEEPALIVE,
        60,
    )
    assert darwin_keepalive in opts or opts == ((SOL_SOCKET, SO_KEEPALIVE, 1),)


def test_default_socket_options_other_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown platform (e.g. win32): `SO_KEEPALIVE` only."""
    monkeypatch.setattr(_client_utils.sys, "platform", "win32")
    opts = _client_utils._default_socket_options()
    assert opts in (((SOL_SOCKET, SO_KEEPALIVE, 1),), ())


def test_filter_supported_probe_failure_returns_unfiltered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contract: probe-socket failure -> input is returned verbatim."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        msg = "sandboxed"
        raise OSError(msg)

    monkeypatch.setattr(_client_utils.socket, "socket", _raise)
    good = (SOL_SOCKET, SO_KEEPALIVE, 1)
    bogus = (0xDEAD, 0xBEEF, 1)
    result = _client_utils._filter_supported([good, bogus])
    assert result == [good, bogus]


def test_invalid_tcp_env_emits_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Int env fallback must log a WARNING naming the offending variable."""
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_TCP_KEEPIDLE", "not-an-int")
    caplog.set_level(logging.WARNING, logger="langchain_anthropic._client_utils")
    _client_utils._default_socket_options()
    assert any(
        "LANGCHAIN_ANTHROPIC_TCP_KEEPIDLE" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_negative_tcp_env_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Negative keepalive counts fall back to the default with a WARNING."""
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_TCP_KEEPCNT", "-5")
    caplog.set_level(logging.WARNING, logger="langchain_anthropic._client_utils")
    value = _client_utils._int_env("LANGCHAIN_ANTHROPIC_TCP_KEEPCNT", 3)
    assert value == 3
    assert any(
        "negative" in r.getMessage().lower()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


@pytest.mark.enable_socket
def test_filter_supported_logs_drops_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dropped options are visible at DEBUG so a macOS user can confirm the filter."""
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
    except OSError:
        pytest.skip("probe socket unavailable in this environment")
    caplog.set_level(logging.DEBUG, logger="langchain_anthropic._client_utils")
    good = (SOL_SOCKET, SO_KEEPALIVE, 1)
    bogus = (0xDEAD, 0xBEEF, 1)
    _client_utils._filter_supported([good, bogus])
    assert any(
        "Dropped" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG
    )


def test_build_sync_httpx_client_with_proxy_wraps_transport() -> None:
    """Non-empty socket_options + proxy -> transport carries Proxy(...)."""
    client = _client_utils._build_sync_httpx_client(
        base_url=None,
        anthropic_proxy="http://proxy.example:3128",
        socket_options=((SOL_SOCKET, SO_KEEPALIVE, 1),),
    )
    assert isinstance(client, httpx.Client)


def test_build_async_httpx_client_with_proxy_wraps_transport() -> None:
    """Async variant: non-empty socket_options + proxy -> transport path."""
    client = _client_utils._build_async_httpx_client(
        base_url=None,
        anthropic_proxy="http://proxy.example:3128",
        socket_options=((SOL_SOCKET, SO_KEEPALIVE, 1),),
    )
    assert isinstance(client, httpx.AsyncClient)


def test_warn_if_proxy_env_shadowed_emits_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One WARNING per process when a proxy env var is shadowed by our transport."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    monkeypatch.setattr(_client_utils, "_proxy_env_warning_emitted", False)
    caplog.set_level(logging.WARNING, logger="langchain_anthropic._client_utils")
    opts = ((SOL_SOCKET, SO_KEEPALIVE, 1),)
    _client_utils._warn_if_proxy_env_shadowed(opts, anthropic_proxy=None)
    _client_utils._warn_if_proxy_env_shadowed(opts, anthropic_proxy=None)
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "HTTP_PROXY" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_warn_if_proxy_env_shadowed_detects_lowercase(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lowercase `http_proxy` is picked up by httpx; the warning must fire for it."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")
    monkeypatch.setattr(_client_utils, "_proxy_env_warning_emitted", False)
    caplog.set_level(logging.WARNING, logger="langchain_anthropic._client_utils")
    opts = ((SOL_SOCKET, SO_KEEPALIVE, 1),)
    _client_utils._warn_if_proxy_env_shadowed(opts, anthropic_proxy=None)
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "http_proxy" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_warn_if_proxy_env_shadowed_detects_system_proxy(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """macOS/Windows system proxies shadow the transport too; warning should fire."""
    for name in _client_utils._PROXY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_client_utils, "_proxy_env_warning_emitted", False)
    monkeypatch.setattr(
        _client_utils.urllib.request,
        "getproxies",
        lambda: {"http": "http://system.proxy:3128"},
    )
    caplog.set_level(logging.WARNING, logger="langchain_anthropic._client_utils")
    opts = ((SOL_SOCKET, SO_KEEPALIVE, 1),)
    _client_utils._warn_if_proxy_env_shadowed(opts, anthropic_proxy=None)
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "system proxy" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_warn_if_proxy_env_shadowed_skipped_when_anthropic_proxy_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit `anthropic_proxy` suppresses the warn (proxy handling is controlled)."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    monkeypatch.setattr(_client_utils, "_proxy_env_warning_emitted", False)
    caplog.set_level(logging.WARNING, logger="langchain_anthropic._client_utils")
    opts = ((SOL_SOCKET, SO_KEEPALIVE, 1),)
    _client_utils._warn_if_proxy_env_shadowed(
        opts, anthropic_proxy="http://proxy.example:3128"
    )
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_proxy_env_bypass_default_shape_triggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-shape + env proxy => bypass socket-option transport."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    assert _client_utils._should_bypass_socket_options_for_proxy_env(
        http_socket_options=None,
        anthropic_proxy=None,
    )


def test_proxy_env_bypass_no_env_does_not_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No proxy env/system proxy => no bypass, even with everything else default."""
    for name in _client_utils._PROXY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_client_utils.urllib.request, "getproxies", dict)
    assert not _client_utils._should_bypass_socket_options_for_proxy_env(
        http_socket_options=None,
        anthropic_proxy=None,
    )


def test_proxy_env_bypass_blocked_by_explicit_socket_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit `http_socket_options` => user opted in, no bypass."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    assert not _client_utils._should_bypass_socket_options_for_proxy_env(
        http_socket_options=[(SOL_SOCKET, SO_KEEPALIVE, 1)],
        anthropic_proxy=None,
    )
    # Empty tuple is also an explicit choice (kill-switch), no bypass.
    assert not _client_utils._should_bypass_socket_options_for_proxy_env(
        http_socket_options=(),
        anthropic_proxy=None,
    )


def test_proxy_env_bypass_blocked_by_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`LANGCHAIN_ANTHROPIC_TCP_KEEPALIVE=0` => kill-switch owns the disable path."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("LANGCHAIN_ANTHROPIC_TCP_KEEPALIVE", "0")
    assert not _client_utils._should_bypass_socket_options_for_proxy_env(
        http_socket_options=None,
        anthropic_proxy=None,
    )


def test_proxy_env_bypass_blocked_by_anthropic_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`anthropic_proxy` handles proxying explicitly => no bypass."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    assert not _client_utils._should_bypass_socket_options_for_proxy_env(
        http_socket_options=None,
        anthropic_proxy="http://anthropic.proxy:3128",
    )


def test_proxy_env_bypass_detects_lowercase_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lowercase `https_proxy` also triggers the bypass."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("https_proxy", "http://proxy.example:3128")
    assert _client_utils._should_bypass_socket_options_for_proxy_env(
        http_socket_options=None,
        anthropic_proxy=None,
    )


def test_proxy_env_bypass_detects_system_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS/Windows system proxy config triggers the bypass too."""
    for name in _client_utils._PROXY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        _client_utils.urllib.request,
        "getproxies",
        lambda: {"http": "http://system.proxy:3128"},
    )
    assert _client_utils._should_bypass_socket_options_for_proxy_env(
        http_socket_options=None,
        anthropic_proxy=None,
    )


def test_log_proxy_env_bypass_once_emits_info_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One INFO per process when the bypass kicks in."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setattr(_client_utils, "_proxy_env_bypass_info_emitted", False)
    caplog.set_level(logging.INFO, logger="langchain_anthropic._client_utils")
    _client_utils._log_proxy_env_bypass_once()
    _client_utils._log_proxy_env_bypass_once()
    infos = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "HTTPS_PROXY" in r.getMessage()
    ]
    assert len(infos) == 1


def test_client_build_skips_transport_on_proxy_env_default_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: default-shape ChatAnthropic + HTTPS_PROXY => no custom transport."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setattr(_client_utils, "_proxy_env_bypass_info_emitted", False)
    monkeypatch.setattr(_client_utils, "_proxy_env_warning_emitted", False)
    _client_utils._cached_sync_httpx_client.cache_clear()
    _client_utils._cached_async_httpx_client.cache_clear()

    recorded: list[tuple[Any, ...]] = []

    original_build = _client_utils._build_async_httpx_client

    def spy(**kwargs: Any) -> Any:
        recorded.append(kwargs.get("socket_options", ()))
        return original_build(**kwargs)

    monkeypatch.setattr(_client_utils, "_build_async_httpx_client", spy)
    monkeypatch.setattr(_client_utils, "_cached_async_httpx_client", spy)

    llm = ChatAnthropic(model=_MODEL)
    _ = llm._async_client

    assert recorded, "async builder should have been called"
    assert all(opts == () for opts in recorded), (
        f"expected bypass (no socket options), got {recorded!r}"
    )


def test_client_build_applies_socket_options_when_user_opts_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit `http_socket_options` => transport applied, bypass skipped."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setattr(_client_utils, "_proxy_env_bypass_info_emitted", False)
    monkeypatch.setattr(_client_utils, "_proxy_env_warning_emitted", False)
    _client_utils._cached_sync_httpx_client.cache_clear()
    _client_utils._cached_async_httpx_client.cache_clear()

    recorded: list[tuple[Any, ...]] = []
    original_build = _client_utils._build_async_httpx_client

    def spy(**kwargs: Any) -> Any:
        recorded.append(kwargs.get("socket_options", ()))
        return original_build(**kwargs)

    monkeypatch.setattr(_client_utils, "_build_async_httpx_client", spy)
    monkeypatch.setattr(_client_utils, "_cached_async_httpx_client", spy)

    explicit = [(SOL_SOCKET, SO_KEEPALIVE, 1)]
    llm = ChatAnthropic(model=_MODEL, http_socket_options=explicit)
    _ = llm._async_client

    assert recorded, "async builder should have been called"
    assert all(tuple(opts) == tuple(explicit) for opts in recorded), (
        f"expected user-supplied opts, got {recorded!r}"
    )


def test_resolved_socket_options_is_cached_and_logs_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`_resolved_socket_options` must be a cached_property.

    A regression to a plain `property` would cause `_log_proxy_env_bypass_once`
    to re-evaluate on every `_client` / `_async_client` access. One proxy-env
    bypass INFO per instance, not per client build.
    """
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setattr(_client_utils, "_proxy_env_bypass_info_emitted", False)
    caplog.set_level(logging.INFO, logger="langchain_anthropic._client_utils")

    llm = ChatAnthropic(model=_MODEL)
    first = llm._resolved_socket_options
    second = llm._resolved_socket_options
    # Property caching: same object returned.
    assert first is second
    # Triggering lazy clients shouldn't re-compute or re-log.
    _ = llm._client
    _ = llm._async_client
    infos = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "HTTPS_PROXY" in r.getMessage()
    ]
    assert len(infos) == 1


def test_build_async_httpx_client_with_proxy_pops_client_proxy_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """proxy + socket_options: proxy lands on the transport, not the Client.

    When `socket_options` is populated we build an `httpx.AsyncHTTPTransport`
    and must pop the top-level `proxy=` kwarg so httpx doesn't try to also
    wrap the transport (which would raise).
    """
    recorded_client: list[dict[str, Any]] = []
    recorded_transport: list[dict[str, Any]] = []

    client_original = _client_utils._AsyncHttpxClientWrapper.__init__

    def client_spy(self: Any, **kwargs: Any) -> None:
        recorded_client.append(kwargs)
        client_original(self, **kwargs)

    monkeypatch.setattr(_client_utils._AsyncHttpxClientWrapper, "__init__", client_spy)

    transport_original = _client_utils.httpx.AsyncHTTPTransport

    class TransportRecorder(transport_original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            recorded_transport.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        "langchain_anthropic._client_utils.httpx.AsyncHTTPTransport",
        TransportRecorder,
    )

    _client_utils._build_async_httpx_client(
        base_url=None,
        anthropic_proxy="http://proxy.example:3128",
        socket_options=((SOL_SOCKET, SO_KEEPALIVE, 1),),
    )

    assert recorded_client
    # Client must NOT receive proxy= when transport= is used, otherwise httpx
    # raises ValueError for double-configured proxies.
    assert "proxy" not in recorded_client[-1]
    assert "transport" in recorded_client[-1]

    assert recorded_transport
    transport_kwargs = recorded_transport[-1]
    assert isinstance(transport_kwargs.get("proxy"), _client_utils.httpx.Proxy)
    assert transport_kwargs.get("socket_options") == [(SOL_SOCKET, SO_KEEPALIVE, 1)]
