"""UUID utility functions."""

from __future__ import annotations

import os
import threading
import time
import typing
from uuid import UUID

try:
    from uuid_utils.compat import uuid7 as _uuid_utils_uuid7

    _HAS_UUID_UTILS = True
except ImportError:
    _HAS_UUID_UTILS = False

_NANOS_PER_SECOND: typing.Final = 1_000_000_000
_MAX_COUNTER: typing.Final = 0x3FFFFFFFFFF

_lock: threading.Lock = threading.Lock()
_last_ms: int = 0
_counter: int = 0


def _new_counter() -> int:
    """Return a fresh random 42-bit counter value with the MSB cleared."""
    return int.from_bytes(os.urandom(6), "big") & 0x1FFFFFFFFFF


def _uuid7_python(ms: int) -> UUID:
    """Generate a UUIDv7 using a pure-Python implementation.

    Args:
        ms: Unix timestamp in milliseconds.

    Returns:
        A UUIDv7 object.
    """
    global _last_ms, _counter  # noqa: PLW0603

    with _lock:
        if ms > _last_ms:
            _last_ms = ms
            _counter = _new_counter()
        else:
            ms = _last_ms
            _counter += 1
            if _counter > _MAX_COUNTER:
                _last_ms += 1
                ms = _last_ms
                _counter = _new_counter()

        current_ms = ms
        current_counter = _counter

    random_tail = int.from_bytes(os.urandom(4), "big")
    counter_hi = (current_counter >> 30) & 0xFFF
    counter_lo = current_counter & 0x3FFFFFFF

    uuid_int = (
        (current_ms & 0xFFFFFFFFFFFF) << 80
        | 0x7 << 76
        | counter_hi << 64
        | 0b10 << 62
        | counter_lo << 32
        | random_tail
    )
    return UUID(int=uuid_int)


def _to_timestamp_and_nanos(nanoseconds: int) -> tuple[int, int]:
    """Split a nanosecond timestamp into seconds and remaining nanoseconds."""
    seconds, nanos = divmod(nanoseconds, _NANOS_PER_SECOND)
    return seconds, nanos


def uuid7(nanoseconds: int | None = None) -> UUID:
    """Generate a UUID from a Unix timestamp in nanoseconds and random bits.

    UUIDv7 objects feature monotonicity within a millisecond.

    Uses ``uuid-utils`` when available, falling back to a pure-Python
    implementation for environments where compiled extensions are unavailable
    (e.g. Pyodide/WebAssembly).

    Args:
        nanoseconds: Optional ns timestamp. If not provided, uses current time.

    Returns:
        A UUIDv7 object.
    """
    if _HAS_UUID_UTILS:
        if nanoseconds is None:
            return _uuid_utils_uuid7()
        seconds, nanos = _to_timestamp_and_nanos(nanoseconds)
        return _uuid_utils_uuid7(timestamp=seconds, nanos=nanos)

    ms = (
        time.time_ns() // 1_000_000 if nanoseconds is None else nanoseconds // 1_000_000
    )
    return _uuid7_python(ms)


__all__ = ["uuid7"]
