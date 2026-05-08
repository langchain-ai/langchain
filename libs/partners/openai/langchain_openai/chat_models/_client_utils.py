"""Helpers for OpenAI httpx client construction, transport tuning, and streaming.

Covers cached default client builders, proxy-aware variants for the
`openai_proxy` path, kernel-level TCP keepalive / `TCP_USER_TIMEOUT` socket
options, and the `_astream_with_chunk_timeout` wrapper that bounds per-chunk
wall-clock time on async SSE streams.

Client-builder boilerplate mirrors the patterns in `openai._base_client`;
socket-option tuning and the streaming timeout are original to this module.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
import sys
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from functools import lru_cache
from typing import Any, TypeVar, cast

import httpx
import openai
from pydantic import SecretStr

logger = logging.getLogger(__name__)

SocketOption = tuple[int, int, int]

# socket.TCP_KEEPIDLE etc. are absent on darwin/win32; use raw UAPI constants.
_LINUX_TCP_KEEPIDLE = 4
_LINUX_TCP_KEEPINTVL = 5
_LINUX_TCP_KEEPCNT = 6
_LINUX_TCP_USER_TIMEOUT = 18

# macOS: same semantics, different constants from <netinet/tcp.h>.
_DARWIN_TCP_KEEPALIVE = 0x10  # idle seconds before first probe
_DARWIN_TCP_KEEPINTVL = 0x101
_DARWIN_TCP_KEEPCNT = 0x102

# Mirrors the openai SDK's pool defaults. Hardcoded to avoid depending on
# an internal module path (openai._constants) that can move across SDK versions.
_DEFAULT_CONNECTION_LIMITS = httpx.Limits(
    max_connections=1000,
    max_keepalive_connections=100,
    keepalive_expiry=5.0,
)


def _int_env(name: str, default: int, *, allow_negative: bool = False) -> int:
    """Read an int env var with graceful fallback + discoverable warning.

    Unparseable or (by default) negative values fall back to `default` and
    emit a single `WARNING` naming the offending variable. A misconfigured
    environment still loads, but operators see the fallback in their logs
    rather than silently getting a surprising default.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s=%r (not an int); falling back to %d.",
            name,
            raw,
            default,
        )
        return default
    if not allow_negative and value < 0:
        logger.warning(
            "Invalid value for %s=%r (negative); falling back to %d.",
            name,
            raw,
            default,
        )
        return default
    return value


def _float_env(name: str, default: float, *, allow_negative: bool = False) -> float:
    """Read a float env var with graceful fallback + discoverable warning.

    See `_int_env`. Negative values are rejected by default so a typo in
    `LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S=-10` can't silently disable the
    wrapper it was meant to configure.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s=%r (not a float); falling back to %s.",
            name,
            raw,
            default,
        )
        return default
    if not allow_negative and value < 0:
        logger.warning(
            "Invalid value for %s=%r (negative); falling back to %s.",
            name,
            raw,
            default,
        )
        return default
    return value


def _filter_supported(opts: list[SocketOption]) -> list[SocketOption]:
    """Drop socket options the running platform rejects.

    Probes each option against a throwaway socket via `setsockopt` and keeps
    only those the kernel accepts. This keeps the library-computed defaults
    non-fatal across platforms that don't implement every Linux option —
    `TCP_USER_TIMEOUT` in particular is Linux-only and silently missing on
    macOS, some minimal kernels, and older gVisor builds. Dropped options
    are logged at `DEBUG` so an operator can confirm whether a kernel-level
    knob took effect on their platform.

    If the probe socket cannot be created (sandboxed runtimes, `pytest-socket`
    under `--disable-socket`, tight seccomp policies), the input list is
    returned unfiltered. This preserves the pass-through behavior used for
    explicit user overrides: unsupported options will surface as a clear
    `OSError` at the first real `connect()` rather than being silently
    dropped during `ChatOpenAI` construction.
    """
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except Exception:
        # Broad catch is deliberate: `pytest_socket` under `--disable-socket`
        # raises `SocketBlockedError` (a `RuntimeError`, not `OSError`), and
        # seccomp/sandboxed runtimes have been observed to raise other
        # `OSError` subclasses and `PermissionError`. The intent is "any
        # inability to create a probe socket -> pass through unfiltered,"
        # and narrowing the type would silently regress sandboxed CI.
        return list(opts)
    try:
        supported: list[SocketOption] = []
        dropped: list[SocketOption] = []
        for level, optname, optval in opts:
            try:
                probe.setsockopt(level, optname, optval)
            except OSError:
                dropped.append((level, optname, optval))
                continue
            supported.append((level, optname, optval))
        if dropped:
            logger.debug(
                "Dropped %d unsupported socket option(s) on %s: %s",
                len(dropped),
                sys.platform,
                dropped,
            )
        return supported
    finally:
        probe.close()


def _default_socket_options() -> tuple[SocketOption, ...]:
    """Return default TCP socket options, or `()` if disabled via env.

    Always returns a tuple (never None) so callers and `@lru_cache` keys
    remain uniform: `()` is the single shape for "no options".

    Target behavior on Linux/gVisor with the full option set: silent peers
    are surfaced within ~90-120s via `SO_KEEPALIVE` + `TCP_USER_TIMEOUT`
    (keepalive path gives a ~90s floor at the defaults; `TCP_USER_TIMEOUT`
    caps at 120s). On platforms that reject some options,
    `_filter_supported` drops them and the bound degrades to whatever the
    remaining options provide.
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


_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_proxy_env_warning_emitted = False
_proxy_env_bypass_info_emitted = False


def _proxy_env_detected() -> bool:
    """True when httpx would pick up a proxy from env or system config.

    Mirrors the surface httpx reads (`urllib.request.getproxies()` plus the
    uppercase env var names) so a positive result means env-proxy
    auto-detection is live on pre-PR code paths.
    """
    if any(os.environ.get(name) for name in _PROXY_ENV_VARS):
        return True
    try:
        return bool(urllib.request.getproxies())
    except Exception:
        return False


def _should_bypass_socket_options_for_proxy_env(
    *,
    http_socket_options: Sequence[SocketOption] | None,
    http_client: Any,
    http_async_client: Any,
    openai_proxy: str | None,
) -> bool:
    """True when default shape + env proxy detected → skip transport injection.

    Preserves pre-PR behavior for apps relying on httpx's env-proxy
    auto-detection. Only triggers when the user has made no explicit choice
    that would signal they want the custom transport:

    - `http_socket_options` left at `None` (default, not `()` or a sequence)
    - `LANGCHAIN_OPENAI_TCP_KEEPALIVE` is not `0` (kill-switch is its own path)
    - No `http_client` or `http_async_client` supplied
    - No `openai_proxy` supplied
    - A proxy env var / system proxy is visible to httpx

    If any of those are set, the user has opted in to the transport path
    (directly or via `openai_proxy`) and normal behavior — including the
    shadowed-proxy WARNING — applies. When the kill-switch is set,
    `_default_socket_options` already returns `()`, so the bypass INFO
    would be noise; route through the normal path instead.
    """
    if http_socket_options is not None:
        return False
    if os.environ.get("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "1") == "0":
        return False
    if http_client is not None or http_async_client is not None:
        return False
    if openai_proxy:
        return False
    return _proxy_env_detected()


def _log_proxy_env_bypass_once() -> None:
    """Emit a one-time INFO when the proxy-env bypass triggers.

    Visibility for operators running with a custom log pipeline: the bypass
    is the *safe* outcome (env-proxy auto-detection preserved), but it means
    socket-level keepalive / `TCP_USER_TIMEOUT` aren't applied on this
    instance. INFO-level, since it's not a problem — just a diagnostic.
    """
    global _proxy_env_bypass_info_emitted
    if _proxy_env_bypass_info_emitted:
        return
    _proxy_env_bypass_info_emitted = True
    active = [name for name in _PROXY_ENV_VARS if os.environ.get(name)]
    source = ", ".join(active) if active else "system proxy configuration"
    logger.info(
        "langchain-openai detected %s and no explicit `http_socket_options` / "
        "`http_client` / `http_async_client` / `openai_proxy`; skipping the "
        "custom `httpx` transport so httpx's env-proxy auto-detection applies. "
        "Pass `http_socket_options=[...]` to opt back into kernel-level TCP "
        "keepalive tuning on top of the env proxy.",
        source,
    )


def _warn_if_proxy_env_shadowed(
    socket_options: tuple[SocketOption, ...],
    *,
    openai_proxy: str | None,
) -> None:
    """Warn once if a custom transport will shadow httpx's proxy auto-detection.

    When `socket_options` is non-empty we pass a custom `httpx` transport,
    which disables httpx's native proxy auto-detection — both the uppercase
    `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` env vars and their lowercase
    equivalents, plus macOS/Windows system proxy config. If the user
    supplies `openai_proxy` explicitly we route through it and the env-var
    handling is moot. Otherwise, a user whose app was transparently relying
    on any of those sources will silently stop using them on upgrade —
    emit a single WARNING so the behavior change is discoverable.

    Detection uses `urllib.request.getproxies()` — the same surface httpx
    reads — so lowercase env vars and macOS/Windows system proxy settings
    are caught alongside the uppercase names.
    """
    global _proxy_env_warning_emitted
    if _proxy_env_warning_emitted or not socket_options or openai_proxy:
        return
    active = [name for name in _PROXY_ENV_VARS if os.environ.get(name)]
    try:
        detected = bool(urllib.request.getproxies())
    except Exception:
        detected = False
    if not active and not detected:
        return
    _proxy_env_warning_emitted = True
    if active:
        source = ", ".join(active) + " set in environment"
    else:
        source = "system proxy configuration detected"
    logger.warning(
        "langchain-openai injected a custom httpx transport to apply "
        "`http_socket_options`, which disables httpx's proxy "
        "auto-detection (%s). Set "
        "`LANGCHAIN_OPENAI_TCP_KEEPALIVE=0` or pass `http_socket_options=()` "
        "to restore default proxy behavior, or supply `openai_proxy` / your "
        "own `http_client` / `http_async_client` to take full control.",
        source,
    )


def _resolve_socket_options(
    value: Sequence[SocketOption] | None,
) -> tuple[SocketOption, ...]:
    """Normalize the user-facing field to the tuple form builders expect.

    - `None` => env-driven defaults (may itself be `()` if the user set
        `LANGCHAIN_OPENAI_TCP_KEEPALIVE=0`). This path runs through
        `_filter_supported()` inside `_default_socket_options()` because
        the library-computed option set is aspirational and silent degradation
        is the right posture.
    - Any other sequence (including empty) => retupled for cache hashability.
        An empty tuple is the explicit "disabled" signal. A non-empty sequence
        is passed verbatim — **not** filtered. The user chose these options
        explicitly, so an unsupported constant should surface as a clear
        `OSError` at connect time, not be silently dropped.

    Always returns a tuple — never `None` — so downstream signatures take
    `tuple[SocketOption, ...]` with `()` as the single "no options" shape.
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
        # httpx ignores limits= when transport= is provided; set it explicitly
        # on the transport to avoid silently shrinking the connection pool.
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
        # See _build_sync_httpx_client for the limits= rationale.
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
    """httpx.Client for the openai_proxy code path.

    When socket options are disabled (`()`), returns a plain
    `httpx.Client(proxy=..., verify=...)` with no transport injected.
    """
    if not socket_options:
        return httpx.Client(proxy=proxy, verify=verify)
    # Mount under `all://` (not `transport=`) so `Client._mounts` mirrors the
    # shape produced by httpx's own `proxy=` path — a single-entry dict keyed
    # by `URLPattern("all://")`. Callers (and the existing proxy integration
    # test) reach into `_mounts` to introspect the proxy URL; a bare
    # `transport=` leaves `_mounts` empty.
    #
    # `httpx.HTTPTransport(proxy=...)` is stricter about string coercion than
    # `httpx.Client(proxy=...)`; wrap in the public `httpx.Proxy` type for
    # version-stable behavior.
    transport = httpx.HTTPTransport(
        proxy=httpx.Proxy(proxy),
        verify=verify,
        socket_options=list(socket_options),
        limits=_DEFAULT_CONNECTION_LIMITS,
    )
    return httpx.Client(mounts={"all://": transport})


def _build_proxied_async_httpx_client(
    proxy: str,
    verify: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> httpx.AsyncClient:
    """httpx.AsyncClient for the openai_proxy code path.

    See `_build_proxied_sync_httpx_client` for the opt-out fallback,
    the `mounts={"all://": ...}` shape, and the `httpx.Proxy` wrapping
    rationale.
    """
    if not socket_options:
        return httpx.AsyncClient(proxy=proxy, verify=verify)
    transport = httpx.AsyncHTTPTransport(
        proxy=httpx.Proxy(proxy),
        verify=verify,
        socket_options=list(socket_options),
        limits=_DEFAULT_CONNECTION_LIMITS,
    )
    return httpx.AsyncClient(mounts={"all://": transport})


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


T = TypeVar("T")

# On Python ≤3.10, asyncio.TimeoutError and builtins.TimeoutError are distinct
# hierarchies, so subclassing only asyncio.TimeoutError would not be caught by
# `except TimeoutError:`. On Python ≥3.11 they are the same object, so listing
# both bases would raise TypeError: duplicate base class. We resolve this at
# class-definition time.
_StreamChunkTimeoutBases: tuple[type, ...] = (
    (asyncio.TimeoutError,)
    if issubclass(asyncio.TimeoutError, TimeoutError)
    else (asyncio.TimeoutError, TimeoutError)
)


class StreamChunkTimeoutError(*_StreamChunkTimeoutBases):  # type: ignore[misc]
    """Raised when no streaming chunk arrives within `stream_chunk_timeout`.

    `issubclass(StreamChunkTimeoutError, asyncio.TimeoutError)` and
    `issubclass(StreamChunkTimeoutError, TimeoutError)` both hold on all
    supported Python versions, so existing `except asyncio.TimeoutError:`
    and `except TimeoutError:` handlers keep catching the exception. On
    Python 3.11+ the two exceptions are the same object, so only
    `asyncio.TimeoutError` appears in `__bases__`.

    Structured attributes (`timeout_s`, `model_name`, `chunks_received`)
    mirror the WARNING log's `extra=` payload so diagnostic code doesn't
    need to regex the message.
    """

    def __init__(
        self,
        timeout_s: float,
        *,
        model_name: str | None = None,
        chunks_received: int = 0,
    ) -> None:
        self.timeout_s = timeout_s
        self.model_name = model_name
        self.chunks_received = chunks_received
        context = []
        if model_name:
            context.append(f"model={model_name}")
        context.append(f"chunks_received={chunks_received}")
        suffix = f" ({', '.join(context)})"
        super().__init__(
            f"No streaming chunk received for {timeout_s:.1f}s{suffix}. The "
            f"connection may be alive at the TCP layer but is not producing "
            f"content. Tune or disable via the `stream_chunk_timeout` "
            f"constructor kwarg (set to None or 0 to disable) or the "
            f"`LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` env var. See also "
            f"`http_socket_options` for the kernel-level TCP timeout that "
            f"catches dead TCP peers."
        )


async def _astream_with_chunk_timeout(
    source: AsyncIterator[T],
    timeout: float | None,
    *,
    model_name: str | None = None,
) -> AsyncIterator[T]:
    """Yield from `source` but bound the per-chunk wait time.

    If `timeout` is None or <=0, yields directly with no wall-clock bound.
    Otherwise, each `__anext__` is wrapped in
    `asyncio.wait_for(..., timeout)`. A timeout raises
    `StreamChunkTimeoutError` (a `TimeoutError` subclass) whose message
    names the knob, the env-var override, the model, and how many chunks
    were received before the stall. A single-line structured log also
    fires at WARNING so the signal is visible in aggregate logging systems
    even when the exception is caught upstream.

    When the timeout is active, the source iterator is explicitly
    `aclose()`-d on early exit (timeout, consumer break, any exception) so
    the underlying httpx streaming connection is released promptly. The
    pass-through branch (timeout disabled) relies on httpx's GC-driven
    cleanup instead — matching the behavior of unwrapped streams.
    """
    if not timeout or timeout <= 0:
        async for item in source:
            yield item
        return

    chunks_received = 0
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
                        "model_name": model_name,
                        "chunks_received": chunks_received,
                    },
                )
                raise StreamChunkTimeoutError(
                    timeout,
                    model_name=model_name,
                    chunks_received=chunks_received,
                ) from e
            chunks_received += 1
            yield chunk
    finally:
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception as cleanup_exc:
                # Best-effort cleanup; don't mask the original exception,
                # but leave a DEBUG trace so pool/transport bugs stay
                # discoverable at the right log level.
                logger.debug(
                    "aclose() during _astream_with_chunk_timeout cleanup "
                    "raised; ignoring",
                    exc_info=cleanup_exc,
                )
