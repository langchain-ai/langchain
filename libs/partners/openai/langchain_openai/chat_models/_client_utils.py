"""Helpers for creating OpenAI API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatOpenAI.

Logic is largely replicated from openai._base_client.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from functools import lru_cache
from typing import Any, TypeVar, cast

import httpx
import openai
from pydantic import SecretStr

logger = logging.getLogger(__name__)

SocketOption = tuple[int, int, int]

# Linux TCP-level constants. `socket.TCP_KEEPIDLE` etc. exist on Linux builds
# of Python but not on darwin/win32 builds, so we reference the raw protocol
# numbers directly to keep the module import-safe across platforms. These
# values are the Linux kernel UAPI constants from `linux/tcp.h` and have been
# stable for over a decade.
_LINUX_TCP_KEEPIDLE = 4
_LINUX_TCP_KEEPINTVL = 5
_LINUX_TCP_KEEPCNT = 6
_LINUX_TCP_USER_TIMEOUT = 18

# macOS: same semantics, different constants from <netinet/tcp.h>.
_DARWIN_TCP_KEEPALIVE = 0x10  # idle seconds before first probe
_DARWIN_TCP_KEEPINTVL = 0x101
_DARWIN_TCP_KEEPCNT = 0x102

# Mirrors openai._constants.DEFAULT_CONNECTION_LIMITS as of openai 2.x.
# The openai SDK writes ``httpx.Limits(max_connections=1000,
# max_keepalive_connections=100)`` without an explicit keepalive_expiry,
# which resolves to httpx's default of 5.0s. We mirror that explicitly here
# so that enabling or disabling the kill-switch (LANGCHAIN_OPENAI_TCP_KEEPALIVE=0)
# never silently changes connection pool retention relative to the pre-PR
# openai SDK default. Hardcoded rather than imported to avoid depending on
# an internal module path that moves across SDK versions. If upstream
# changes, revisit here.
_DEFAULT_CONNECTION_LIMITS = httpx.Limits(
    max_connections=1000,
    max_keepalive_connections=100,
    keepalive_expiry=5.0,
)


def _int_env(name: str, default: int) -> int:
    """Read an int env var with graceful fallback on garbage input."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    """Read a float env var with graceful fallback on garbage input."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _filter_supported(opts: list[SocketOption]) -> list[SocketOption]:
    """Drop socket options the running platform rejects.

    Probes each option against a throwaway socket via ``setsockopt`` and keeps
    only those the kernel accepts. This keeps the library-computed defaults
    non-fatal across platforms that don't implement every Linux option —
    ``TCP_USER_TIMEOUT`` in particular is Linux-only and silently missing on
    macOS, some minimal kernels, and older gVisor builds.

    If the probe socket cannot be created (sandboxed runtimes, ``pytest-socket``
    under ``--disable-socket``, tight seccomp policies), the input list is
    returned unfiltered. This preserves the pass-through behavior used for
    explicit user overrides: unsupported options will surface as a clear
    ``OSError`` at the first real ``connect()`` rather than being silently
    dropped during ``ChatOpenAI`` construction.
    """
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except Exception:
        return list(opts)
    try:
        supported: list[SocketOption] = []
        for level, optname, optval in opts:
            try:
                probe.setsockopt(level, optname, optval)
                supported.append((level, optname, optval))
            except OSError:
                # Platform doesn't support this option; silently skip.
                pass
        return supported
    finally:
        probe.close()


def _default_socket_options() -> tuple[SocketOption, ...]:
    """Return default TCP socket options, or ``()`` if disabled via env.

    Always returns a tuple (never None) so callers and ``@lru_cache`` keys
    remain uniform: ``()`` is the single shape for "no options".

    Target behavior on Linux/gVisor with the full option set: silent peers
    are surfaced within ~90-135s via ``SO_KEEPALIVE`` + ``TCP_USER_TIMEOUT``.
    On platforms that reject some options, ``_filter_supported`` drops them
    and the bound degrades to whatever the remaining options provide.
    """
    if os.environ.get("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "1") == "0":
        return ()

    keepidle = _int_env("LANGCHAIN_OPENAI_TCP_KEEPIDLE", 60)
    keepintvl = _int_env("LANGCHAIN_OPENAI_TCP_KEEPINTVL", 10)
    keepcnt = _int_env("LANGCHAIN_OPENAI_TCP_KEEPCNT", 3)
    user_timeout_ms = _int_env("LANGCHAIN_OPENAI_TCP_USER_TIMEOUT_MS", 120000)

    opts: list[SocketOption] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if sys.platform == "linux":
        opts += [
            (socket.IPPROTO_TCP, _LINUX_TCP_KEEPIDLE, keepidle),
            (socket.IPPROTO_TCP, _LINUX_TCP_KEEPINTVL, keepintvl),
            (socket.IPPROTO_TCP, _LINUX_TCP_KEEPCNT, keepcnt),
            (socket.IPPROTO_TCP, _LINUX_TCP_USER_TIMEOUT, user_timeout_ms),
        ]
    elif sys.platform == "darwin":
        opts += [
            (socket.IPPROTO_TCP, _DARWIN_TCP_KEEPALIVE, keepidle),
            (socket.IPPROTO_TCP, _DARWIN_TCP_KEEPINTVL, keepintvl),
            (socket.IPPROTO_TCP, _DARWIN_TCP_KEEPCNT, keepcnt),
        ]
    # Windows (win32): SO_KEEPALIVE only; per-option tuning requires WSAIoctl.
    return tuple(_filter_supported(opts))


def _resolve_socket_options(
    value: Sequence[SocketOption] | None,
) -> tuple[SocketOption, ...]:
    """Normalise the user-facing field to the tuple form builders expect.

    - ``None`` => env-driven defaults (may itself be ``()`` if the user set
      ``LANGCHAIN_OPENAI_TCP_KEEPALIVE=0``). This path runs through
      ``_filter_supported()`` inside ``_default_socket_options()`` because
      the library-computed option set is aspirational and silent degradation
      is the right posture.
    - Any other sequence (including empty) => retupled for cache hashability.
      An empty tuple is the explicit "disabled" signal. A non-empty sequence
      is passed verbatim — **not** filtered. The user chose these options
      explicitly, so an unsupported constant should surface as a clear
      ``OSError`` at connect time, not be silently dropped.

    Always returns a tuple — never ``None`` — so downstream signatures take
    ``tuple[SocketOption, ...]`` with ``()`` as the single "no options" shape.
    """
    if value is None:
        return _default_socket_options()
    return tuple(value)


class _SyncHttpxClientWrapper(openai.DefaultHttpxClient):
    """Borrowed from openai._base_client."""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncHttpxClientWrapper(openai.DefaultAsyncHttpxClient):
    """Borrowed from openai._base_client."""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass


def _build_sync_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _SyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        "timeout": timeout,
    }
    if socket_options:
        # Client ignores ``limits=`` when a transport is provided, so configure
        # limits explicitly on the transport, mirroring the openai SDK's pool
        # config to avoid a silent 10x downgrade.
        # NB: passing ``transport=`` also disables httpx's env-proxy
        # auto-detection (HTTP_PROXY/HTTPS_PROXY/...). This is a documented
        # known limitation; users relying on env proxies for OpenAI traffic
        # can opt out via LANGCHAIN_OPENAI_TCP_KEEPALIVE=0 or bring their
        # own http_client.
        kwargs["transport"] = httpx.HTTPTransport(
            socket_options=list(socket_options),
            limits=_DEFAULT_CONNECTION_LIMITS,
        )
    return _SyncHttpxClientWrapper(**kwargs)


def _build_async_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _AsyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        "timeout": timeout,
    }
    if socket_options:
        # See _build_sync_httpx_client for the rationale on the explicit
        # limits= and the env-proxy-detection caveat.
        kwargs["transport"] = httpx.AsyncHTTPTransport(
            socket_options=list(socket_options),
            limits=_DEFAULT_CONNECTION_LIMITS,
        )
    return _AsyncHttpxClientWrapper(**kwargs)


def _build_proxied_sync_httpx_client(
    proxy: str,
    verify: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> httpx.Client:
    """httpx.Client for the openai_proxy code path, with P2 applied.

    When the user has explicitly disabled socket options (``()``), we return
    the original ``httpx.Client(proxy=..., verify=...)`` shape — no transport,
    no ``_DEFAULT_CONNECTION_LIMITS`` — so the opt-out is a *strict no-op* on
    the proxy path and pre-PR behavior is byte-identical.
    """
    if not socket_options:
        return httpx.Client(proxy=proxy, verify=verify)
    # httpx.HTTPTransport(proxy=...) is stricter about string coercion than
    # httpx.Client(proxy=...); wrap in the public httpx.Proxy type for
    # version-stable behavior.
    transport = httpx.HTTPTransport(
        proxy=httpx.Proxy(proxy),
        verify=verify,
        socket_options=list(socket_options),
        limits=_DEFAULT_CONNECTION_LIMITS,
    )
    return httpx.Client(transport=transport)


def _build_proxied_async_httpx_client(
    proxy: str,
    verify: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> httpx.AsyncClient:
    """httpx.AsyncClient for the openai_proxy code path, with P2 applied.

    See :func:`_build_proxied_sync_httpx_client` for rationale on the opt-out
    fallback and the ``httpx.Proxy`` wrapping.
    """
    if not socket_options:
        return httpx.AsyncClient(proxy=proxy, verify=verify)
    transport = httpx.AsyncHTTPTransport(
        proxy=httpx.Proxy(proxy),
        verify=verify,
        socket_options=list(socket_options),
        limits=_DEFAULT_CONNECTION_LIMITS,
    )
    return httpx.AsyncClient(transport=transport)


@lru_cache
def _cached_sync_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _SyncHttpxClientWrapper:
    return _build_sync_httpx_client(base_url, timeout, socket_options)


@lru_cache
def _cached_async_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _AsyncHttpxClientWrapper:
    return _build_async_httpx_client(base_url, timeout, socket_options)


def _get_default_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _SyncHttpxClientWrapper:
    """Get default httpx client.

    Uses cached client unless timeout is `httpx.Timeout`, which is not hashable.
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_sync_httpx_client(base_url, timeout, socket_options)
    else:
        return _cached_sync_httpx_client(base_url, timeout, socket_options)


def _get_default_async_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _AsyncHttpxClientWrapper:
    """Get default httpx client.

    Uses cached client unless timeout is `httpx.Timeout`, which is not hashable.
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_async_httpx_client(base_url, timeout, socket_options)
    else:
        return _cached_async_httpx_client(base_url, timeout, socket_options)


def _resolve_sync_and_async_api_keys(
    api_key: SecretStr | Callable[[], str] | Callable[[], Awaitable[str]],
) -> tuple[str | None | Callable[[], str], str | Callable[[], Awaitable[str]]]:
    """Resolve sync and async API key values.

    Because OpenAI and AsyncOpenAI clients support either sync or async callables for
    the API key, we need to resolve separate values here.
    """
    if isinstance(api_key, SecretStr):
        sync_api_key_value: str | None | Callable[[], str] = api_key.get_secret_value()
        async_api_key_value: str | Callable[[], Awaitable[str]] = (
            api_key.get_secret_value()
        )
    elif callable(api_key):
        if inspect.iscoroutinefunction(api_key):
            async_api_key_value = api_key
            sync_api_key_value = None
        else:
            sync_api_key_value = cast(Callable, api_key)

            async def async_api_key_wrapper() -> str:
                return await asyncio.get_event_loop().run_in_executor(
                    None, cast(Callable, api_key)
                )

            async_api_key_value = async_api_key_wrapper

    return sync_api_key_value, async_api_key_value


# ---------------------------------------------------------------------------
# P1 — per-chunk wall-clock timeout for async streaming iterators.
# ---------------------------------------------------------------------------

T = TypeVar("T")


class StreamChunkTimeoutError(asyncio.TimeoutError):
    """Raised when no streaming chunk arrives within ``stream_chunk_timeout``.

    Subclasses ``asyncio.TimeoutError`` so existing ``except TimeoutError:``
    handlers keep working, but carries a structured, self-describing message
    so the first thing an operator sees in logs names the knob and how to
    tune it.
    """


async def _astream_with_chunk_timeout(
    source: AsyncIterator[T],
    timeout: float | None,
) -> AsyncIterator[T]:
    """Yield from ``source`` but bound the per-chunk wait time.

    If ``timeout`` is None or <=0, yields directly with no wall-clock bound.
    Otherwise, each ``__anext__`` is wrapped in
    ``asyncio.wait_for(..., timeout)``. A timeout raises
    ``StreamChunkTimeoutError`` (a ``TimeoutError`` subclass) with a clear
    message naming the knob and the env-var override. A single-line
    structured log also fires at WARNING so the signal is visible in
    aggregate logging systems even when the exception is caught upstream.

    The source iterator is explicitly ``aclose()``-d on early exit (timeout,
    consumer break, any exception) so the underlying httpx streaming
    connection is released promptly rather than left dangling.
    """
    if not timeout or timeout <= 0:
        # Fast path: no wall-clock bound. We don't wrap this in try/finally
        # to aclose() the source — if a consumer breaks early, source falls
        # back on httpx's GC-driven cleanup, which is pre-existing behavior
        # and out of scope for this PR.
        async for item in source:
            yield item
        return

    it = source.__aiter__()
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(it.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as e:
                logger.warning(
                    "langchain_openai.stream_chunk_timeout fired",
                    extra={
                        "source": "stream_chunk_timeout",
                        "timeout_s": timeout,
                    },
                )
                msg = (
                    f"No streaming chunk received for {timeout:.1f}s. The "
                    f"connection may be alive at the TCP layer but is not "
                    f"producing content. Tune or disable via the "
                    f"`stream_chunk_timeout` constructor kwarg (set to None "
                    f"or 0 to disable) or the "
                    f"`LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` env var. See "
                    f"also `http_socket_options` for the kernel-level TCP "
                    f"timeout that catches dead TCP peers."
                )
                raise StreamChunkTimeoutError(msg) from e
            yield chunk
    finally:
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: S110
                # Best-effort cleanup; don't mask the original exception.
                pass
