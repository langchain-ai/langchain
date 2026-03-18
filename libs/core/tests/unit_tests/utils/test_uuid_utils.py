import time
from uuid import UUID

import pytest

from langchain_core.utils.uuid import _uuid7_python, uuid7


def _uuid_v7_ms(uuid_obj: UUID | str) -> int:
    """Extract milliseconds since epoch from a UUIDv7 using string layout.

    UUIDv7 stores Unix time in ms in the first 12 hex chars of the canonical
    string representation (48 msb bits).
    """
    s = str(uuid_obj).replace("-", "")
    return int(s[:12], 16)


def _uuid_v7_version(uuid_obj: UUID | str) -> int:
    """Extract the version nibble (bits 48-51) from a UUID hex string."""
    s = str(uuid_obj).replace("-", "")
    return int(s[12], 16)


def _uuid_v7_variant(uuid_obj: UUID | str) -> int:
    """Extract the two variant bits (bits 64-65) from a UUID hex string."""
    s = str(uuid_obj).replace("-", "")
    return (int(s[16], 16) >> 2) & 0b11


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


# ---------------------------------------------------------------------------
# Tests for the pure-Python fallback (_uuid7_python)
# ---------------------------------------------------------------------------


def test_python_fallback_timestamp() -> None:
    """Pure-Python fallback embeds the correct ms timestamp."""
    ns = time.time_ns()
    ms = ns // 1_000_000
    result = _uuid7_python(ms)
    assert _uuid_v7_ms(result) == ms


def test_python_fallback_version_and_variant() -> None:
    """Pure-Python fallback sets version=7 and RFC 4122 variant bits."""
    result = _uuid7_python(time.time_ns() // 1_000_000)
    assert _uuid_v7_version(result) == 7
    assert _uuid_v7_variant(result) == 0b10


def test_python_fallback_returns_uuid() -> None:
    """Pure-Python fallback returns a UUID instance."""
    result = _uuid7_python(time.time_ns() // 1_000_000)
    assert isinstance(result, UUID)


def test_python_fallback_monotonicity() -> None:
    """Pure-Python fallback UUIDs are monotonically increasing."""
    last = ""
    for n in range(10_000):
        i = str(_uuid7_python(time.time_ns() // 1_000_000))
        if n > 0 and i <= last:
            msg = f"Pure-Python UUIDs are not monotonic: {last} versus {i}"
            raise RuntimeError(msg)
        last = i


def test_python_fallback_same_ms_monotonicity() -> None:
    """Within a fixed millisecond the counter increments produce monotonic UUIDs."""
    fixed_ms = time.time_ns() // 1_000_000
    results = [str(_uuid7_python(fixed_ms)) for _ in range(1_000)]
    assert results == sorted(results), "UUIDs within same ms are not monotonic"


@pytest.mark.parametrize("ms_offset", [-1, 0, 1])
def test_python_fallback_clock_regression(ms_offset: int) -> None:
    """Clock regression is handled gracefully — output is always monotonic."""
    # Generate a baseline UUID first to set _last_ms
    base_ms = time.time_ns() // 1_000_000
    first = str(_uuid7_python(base_ms))
    # Now call with base_ms again (regression / same ms)
    second = str(_uuid7_python(base_ms + ms_offset))
    assert second > first, (
        f"UUID not monotonic after clock regression (offset={ms_offset}): "
        f"{first} vs {second}"
    )
