import time
from uuid import UUID

from langchain_core.utils.uuid import uuid7


def _uuid_v7_ms(uuid_obj: UUID | str) -> int:
    """Extract milliseconds since epoch from a UUIDv7 using string layout.

    UUIDv7 stores Unix time in ms in the first 12 hex chars of the canonical
    string representation (48 msb bits).
    """
    s = str(uuid_obj).replace("-", "")
    return int(s[:12], 16)


def test_uuid7() -> None:
    """Some simple tests."""
    # Note the sequence value increments by 1 between each of these uuid7(...) calls
    ns = time.time_ns()
    ms = ns // 1_000_000
    out1 = str(uuid7(ns))

    # Verify that the timestamp part matches
    out1_ms = _uuid_v7_ms(out1)
    assert out1_ms == ms


def _uuid_v7_sortable_prefix(uuid_obj: UUID) -> int:
    """Extract the timestamp + counter bits (top 96 bits minus version/variant)."""
    val = uuid_obj.int
    timestamp_ms = (val >> 80) & 0xFFFFFFFFFFFF
    counter_hi = (val >> 64) & 0x0FFF
    counter_lo = (val >> 32) & 0x3FFFFFFF
    return (timestamp_ms << 42) | (counter_hi << 30) | counter_lo


def test_monotonicity() -> None:
    """Test that UUIDs are monotonically increasing.

    UUIDv7 guarantees monotonicity of the timestamp+counter portion,
    but the trailing 32-bit random field can cause raw string or int
    comparisons to appear non-monotonic. Compare only the sortable prefix.
    """
    last_prefix = -1
    last_uuid = None
    for n in range(100_000):
        u = uuid7()
        prefix = _uuid_v7_sortable_prefix(u)
        if n > 0 and prefix < last_prefix:
            msg = f"UUIDs are not monotonic: {last_uuid} versus {u}"
            raise RuntimeError(msg)
        last_prefix = prefix
        last_uuid = u
