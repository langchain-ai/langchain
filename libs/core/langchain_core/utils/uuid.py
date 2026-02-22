"""UUID utility functions.

This module exports a uuid7 function to generate monotonic, time-ordered UUIDs
for tracing and similar operations.
"""

from __future__ import annotations

import typing
from uuid import UUID

from uuid_utils.compat import uuid7 as _uuid_utils_uuid7

if typing.TYPE_CHECKING:
    from uuid import UUID

_NANOS_PER_SECOND: typing.Final = 1_000_000_000


def _to_timestamp_and_nanos(nanoseconds: int) -> tuple[int, int]:
    """Split a nanosecond timestamp into seconds and remaining nanoseconds."""
    seconds, nanos = divmod(nanoseconds, _NANOS_PER_SECOND)
    return seconds, nanos


def uuid7(nanoseconds: int | None = None) -> UUID:
    """Generate a UUID from a Unix timestamp in nanoseconds and random bits.

    UUIDv7 objects feature monotonicity within a millisecond.

    Args:
        nanoseconds: Optional ns timestamp. If not provided, uses current time.

    Returns:
        A UUIDv7 object.
    """
    # --- 48 ---   -- 4 --   --- 12 ---   -- 2 --   --- 30 ---   - 32 -
    # unix_ts_ms | version | counter_hi | variant | counter_lo | random
    #
    # 'counter = counter_hi | counter_lo' is a 42-bit counter constructed
    # with Method 1 of RFC 9562, §6.2, and its MSB is set to 0.
    #
    # 'random' is a 32-bit random value regenerated for every new UUID.
    #
    # If multiple UUIDs are generated within the same millisecond, the LSB
    # of 'counter' is incremented by 1. When overflowing, the timestamp is
    # advanced and the counter is reset to a random 42-bit integer with MSB
    # set to 0.

    # For now, just delegate to the uuid_utils implementation
    if nanoseconds is None:
        return _uuid_utils_uuid7()
    seconds, nanos = _to_timestamp_and_nanos(nanoseconds)
    return _uuid_utils_uuid7(timestamp=seconds, nanos=nanos)


__all__ = ["uuid7"]
