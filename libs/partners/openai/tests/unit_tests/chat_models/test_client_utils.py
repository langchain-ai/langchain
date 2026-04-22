"""Unit tests for `langchain_openai.chat_models._client_utils`.

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

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import _client_utils

SOL_SOCKET = socket.SOL_SOCKET
SO_KEEPALIVE = socket.SO_KEEPALIVE


@pytest.fixture(autouse=True)
def _clear_langchain_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure LANGCHAIN_OPENAI_* env vars don't leak between tests."""
    for name in list(os.environ):
        if name.startswith("LANGCHAIN_OPENAI_") or name == "OPENAI_API_KEY":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


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
    monkeypatch.setenv("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")
    opts = _client_utils._default_socket_options()
    assert opts == ()
    assert isinstance(opts, tuple)


@pytest.mark.enable_socket
def test_filter_supported_drops_unsupported() -> None:
    """An option with a deliberately-bogus level should be silently dropped.

    Requires a real probe socket, so opt out of the suite-wide
    `--disable-socket`. If the probe still cannot be created (unusual
    sandboxed runner), the helper falls back to pass-through; assert that
    contract explicitly rather than masking the behavior.
    """
    good = (SOL_SOCKET, SO_KEEPALIVE, 1)
    # Very high level number the kernel will reject.
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
        timeout=None,
        socket_options=((SOL_SOCKET, SO_KEEPALIVE, 1),),
    )
    assert recorded, "expected one call when socket_options populated"
    assert "transport" in recorded[-1]

    recorded.clear()
    _client_utils._build_async_httpx_client(
        base_url=None, timeout=None, socket_options=()
    )
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
        "langchain_openai.chat_models._client_utils.httpx.AsyncHTTPTransport",
        Recorder,
    )

    _client_utils._build_async_httpx_client(
        base_url=None,
        timeout=None,
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
    recorded: list[tuple[str, tuple, tuple]] = []

    def spy_async(
        base_url: str | None,
        timeout: Any,
        socket_options: tuple = (),
    ) -> Any:
        recorded.append(("async", (base_url, timeout), tuple(socket_options)))
        # Return a real (but unused) client so init completes.
        return _client_utils._AsyncHttpxClientWrapper(
            base_url=base_url or "https://api.openai.com/v1", timeout=timeout
        )

    def spy_sync(
        base_url: str | None,
        timeout: Any,
        socket_options: tuple = (),
    ) -> Any:
        recorded.append(("sync", (base_url, timeout), tuple(socket_options)))
        return _client_utils._SyncHttpxClientWrapper(
            base_url=base_url or "https://api.openai.com/v1", timeout=timeout
        )

    monkeypatch.setattr(
        "langchain_openai.chat_models.base._get_default_async_httpx_client",
        spy_async,
    )
    monkeypatch.setattr(
        "langchain_openai.chat_models.base._get_default_httpx_client",
        spy_sync,
    )

    # (1) Unset -> None -> env-driven defaults (non-empty on linux/darwin CI).
    ChatOpenAI(model="gpt-4o")
    assert recorded, "expected a default-client build"
    _, _, opts1 = recorded[-1]
    assert isinstance(opts1, tuple)

    # (2) Explicit empty tuple -> ().
    recorded.clear()
    ChatOpenAI(model="gpt-4o", http_socket_options=())
    assert recorded
    assert all(opts == () for _, _, opts in recorded)

    # (3) Populated sequence -> verbatim passthrough (not filtered).
    recorded.clear()
    ChatOpenAI(
        model="gpt-4o",
        http_socket_options=[(SOL_SOCKET, SO_KEEPALIVE, 1)],
    )
    assert recorded
    for _, _, opts in recorded:
        assert opts == ((SOL_SOCKET, SO_KEEPALIVE, 1),)


def test_openai_proxy_branch_applies_socket_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`openai_proxy` path must go through the socket-options-aware proxied helper."""
    recorded: list[dict[str, Any]] = []

    def spy(proxy: str, verify: Any, socket_options: tuple = ()) -> httpx.AsyncClient:
        recorded.append(
            {"proxy": proxy, "verify": verify, "socket_options": socket_options}
        )
        return httpx.AsyncClient()

    monkeypatch.setattr(
        "langchain_openai.chat_models.base._build_proxied_async_httpx_client",
        spy,
    )
    # Sync branch should also be covered — spy on that too.
    sync_recorded: list[dict[str, Any]] = []

    def sync_spy(proxy: str, verify: Any, socket_options: tuple = ()) -> httpx.Client:
        sync_recorded.append(
            {"proxy": proxy, "verify": verify, "socket_options": socket_options}
        )
        return httpx.Client()

    monkeypatch.setattr(
        "langchain_openai.chat_models.base._build_proxied_sync_httpx_client",
        sync_spy,
    )

    ChatOpenAI(
        model="gpt-4o",
        openai_proxy="http://proxy.example.com:3128",
        http_socket_options=[(SOL_SOCKET, SO_KEEPALIVE, 1)],
    )

    assert recorded, "expected async proxied helper to be called"
    assert recorded[-1]["proxy"] == "http://proxy.example.com:3128"
    assert recorded[-1]["socket_options"] == ((SOL_SOCKET, SO_KEEPALIVE, 1),)

    assert sync_recorded, "expected sync proxied helper to be called"
    assert sync_recorded[-1]["proxy"] == "http://proxy.example.com:3128"
    assert sync_recorded[-1]["socket_options"] == ((SOL_SOCKET, SO_KEEPALIVE, 1),)


def test_user_supplied_http_async_client_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the user passes an http_async_client, we must not mutate it."""
    default_calls: list[Any] = []
    proxied_calls: list[Any] = []

    def default_async_spy(*args: Any, **kwargs: Any) -> Any:
        default_calls.append((args, kwargs))
        msg = "default async builder should not run"
        raise AssertionError(msg)

    def proxied_async_spy(*args: Any, **kwargs: Any) -> Any:
        proxied_calls.append((args, kwargs))
        msg = "proxied async builder should not run"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "langchain_openai.chat_models.base._get_default_async_httpx_client",
        default_async_spy,
    )
    monkeypatch.setattr(
        "langchain_openai.chat_models.base._build_proxied_async_httpx_client",
        proxied_async_spy,
    )

    user_client = httpx.AsyncClient()
    user_sync_client = httpx.Client()

    model = ChatOpenAI(
        model="gpt-4o",
        http_client=user_sync_client,
        http_async_client=user_client,
        http_socket_options=[(SOL_SOCKET, SO_KEEPALIVE, 1)],
    )

    assert default_calls == []
    assert proxied_calls == []
    assert model.http_async_client is user_client
    assert model.http_client is user_sync_client


def test_default_path_opt_out_is_strict_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With LANGCHAIN_OPENAI_TCP_KEEPALIVE=0 we inject no transport.

    Boundary assertion on `_AsyncHttpxClientWrapper.__init__` kwargs — our
    helper passed nothing, so httpx falls back to its own native behavior
    (env-proxy handling, pool defaults, trust_env, etc.) completely
    unaffected by this library.
    """
    monkeypatch.setenv("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")

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

    ChatOpenAI(model="gpt-4o")

    assert recorded_sync, "expected the sync default client to be built"
    assert "transport" not in recorded_sync[-1]
    assert recorded_async, "expected the async default client to be built"
    assert "transport" not in recorded_async[-1]


def test_invalid_env_values_degrade_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    """Garbage in LANGCHAIN_OPENAI_TCP_* env vars must not crash model init."""
    monkeypatch.setenv("LANGCHAIN_OPENAI_TCP_KEEPIDLE", "not-an-int")
    monkeypatch.setenv("LANGCHAIN_OPENAI_TCP_KEEPINTVL", "")
    monkeypatch.setenv("LANGCHAIN_OPENAI_TCP_KEEPCNT", "NaN")
    monkeypatch.setenv("LANGCHAIN_OPENAI_TCP_USER_TIMEOUT_MS", "abc")

    opts = _client_utils._default_socket_options()
    assert isinstance(opts, tuple)
    # Fallback values (60/10/3/120000) are used; on Linux, the full option
    # set should still be present because the fallbacks are valid.
    # (Windows/darwin may filter some options; at minimum SO_KEEPALIVE
    # survives.)
    assert (SOL_SOCKET, SO_KEEPALIVE, 1) in opts

    # Instantiating a model doesn't raise.
    ChatOpenAI(model="gpt-4o")


def test_invalid_stream_chunk_timeout_env_degrades_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garbage in LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S must not crash init."""
    monkeypatch.setenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "not-a-float")
    model = ChatOpenAI(model="gpt-4o")
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


def test_default_socket_options_other_platform(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("LANGCHAIN_OPENAI_TCP_KEEPIDLE", "not-an-int")
    caplog.set_level(
        logging.WARNING, logger="langchain_openai.chat_models._client_utils"
    )
    _client_utils._default_socket_options()
    assert any(
        "LANGCHAIN_OPENAI_TCP_KEEPIDLE" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_negative_tcp_env_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Negative keepalive counts fall back to the default with a WARNING."""
    monkeypatch.setenv("LANGCHAIN_OPENAI_TCP_KEEPCNT", "-5")
    caplog.set_level(
        logging.WARNING, logger="langchain_openai.chat_models._client_utils"
    )
    value = _client_utils._int_env("LANGCHAIN_OPENAI_TCP_KEEPCNT", 3)
    assert value == 3
    assert any(
        "negative" in r.getMessage().lower()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_filter_supported_logs_drops_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dropped options are visible at DEBUG so a macOS user can confirm the filter."""
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
    except OSError:
        pytest.skip("probe socket unavailable in this environment")
    caplog.set_level(logging.DEBUG, logger="langchain_openai.chat_models._client_utils")
    good = (SOL_SOCKET, SO_KEEPALIVE, 1)
    bogus = (0xDEAD, 0xBEEF, 1)
    _client_utils._filter_supported([good, bogus])
    assert any(
        "Dropped" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG
    )


def test_build_proxied_async_httpx_client_opt_out_returns_plain_client() -> None:
    """Empty socket_options -> plain httpx.AsyncClient, no transport injection."""
    client = _client_utils._build_proxied_async_httpx_client(
        proxy="http://proxy.example:3128",
        verify=True,
        socket_options=(),
    )
    assert isinstance(client, httpx.AsyncClient)


def test_build_proxied_async_httpx_client_wraps_transport() -> None:
    """Non-empty socket_options -> real httpx.AsyncHTTPTransport wiring executes.

    Exercises the proxy-wrapping bodies end-to-end so a change to httpx's
    `Proxy`/transport signatures would surface here, not at connect time.
    """
    client = _client_utils._build_proxied_async_httpx_client(
        proxy="http://proxy.example:3128",
        verify=True,
        socket_options=((SOL_SOCKET, SO_KEEPALIVE, 1),),
    )
    assert isinstance(client, httpx.AsyncClient)


def test_build_proxied_sync_httpx_client_opt_out_returns_plain_client() -> None:
    client = _client_utils._build_proxied_sync_httpx_client(
        proxy="http://proxy.example:3128",
        verify=True,
        socket_options=(),
    )
    assert isinstance(client, httpx.Client)


def test_build_proxied_sync_httpx_client_wraps_transport() -> None:
    client = _client_utils._build_proxied_sync_httpx_client(
        proxy="http://proxy.example:3128",
        verify=True,
        socket_options=((SOL_SOCKET, SO_KEEPALIVE, 1),),
    )
    assert isinstance(client, httpx.Client)


def test_warn_if_proxy_env_shadowed_emits_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One WARNING per process when a proxy env var is shadowed by our transport."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    monkeypatch.setattr(_client_utils, "_proxy_env_warning_emitted", False)
    caplog.set_level(
        logging.WARNING, logger="langchain_openai.chat_models._client_utils"
    )
    opts = ((SOL_SOCKET, SO_KEEPALIVE, 1),)
    _client_utils._warn_if_proxy_env_shadowed(opts, openai_proxy=None)
    _client_utils._warn_if_proxy_env_shadowed(opts, openai_proxy=None)
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "HTTP_PROXY" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_warn_if_proxy_env_shadowed_skipped_when_openai_proxy_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit `openai_proxy` suppresses the warn (proxy handling is controlled)."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    monkeypatch.setattr(_client_utils, "_proxy_env_warning_emitted", False)
    caplog.set_level(
        logging.WARNING, logger="langchain_openai.chat_models._client_utils"
    )
    opts = ((SOL_SOCKET, SO_KEEPALIVE, 1),)
    _client_utils._warn_if_proxy_env_shadowed(
        opts, openai_proxy="http://proxy.example:3128"
    )
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
