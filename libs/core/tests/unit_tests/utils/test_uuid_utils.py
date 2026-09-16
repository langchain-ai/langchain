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


def test_monotonicity() -> None:
    """Test that UUIDs are monotonically increasing."""
    last = ""
    for n in range(100_000):
        i = str(uuid7())
        if n > 0 and i <= last:
            msg = f"UUIDs are not monotonic: {last} versus {i}"
            raise RuntimeError(msg)
        last = i


def test_uuid7_version_and_properties() -> None:
    """Test UUIDv7 version and object properties."""
    val = uuid7()
    assert isinstance(val, UUID)
    assert val.version == 7


def test_uuid7_with_specific_timestamp() -> None:
    """Test UUIDv7 with explicit nanosecond timestamp."""
    # Test year 2026 timestamp in nanoseconds
    ns = 1_773_650_000_000_000_000
    ms = ns // 1_000_000
    val = uuid7(ns)
    assert _uuid_v7_ms(val) == ms
    assert val.version == 7
