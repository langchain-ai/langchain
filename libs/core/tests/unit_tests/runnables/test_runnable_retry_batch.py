"""Tests for RunnableRetry.batch/abatch output alignment (issue #35475).

Verifies that batch outputs remain correctly aligned with their input indices
when some items succeed and others fail across retry attempts.
"""

import pytest

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.retry import RunnableRetry


def _make_flaky_runnable(
    fail_always: set[int],
    fail_first_n: dict[int, int] | None = None,
    max_attempts: int = 3,
) -> RunnableRetry:
    """Create a runnable that fails selectively based on input value.

    Args:
        fail_always: Set of input values that always raise ValueError.
        fail_first_n: Dict mapping input value to number of times it should fail
            before succeeding.
        max_attempts: Maximum retry attempts.
    """
    call_counts: dict[int, int] = {}
    fail_first_n = fail_first_n or {}

    def func(x: int) -> int:
        call_counts[x] = call_counts.get(x, 0) + 1
        if x in fail_always:
            msg = f"Always fails for {x}"
            raise ValueError(msg)
        if x in fail_first_n and call_counts[x] <= fail_first_n[x]:
            msg = f"Transient failure for {x}"
            raise ValueError(msg)
        return x * 10

    return RunnableRetry(
        bound=RunnableLambda(func),
        retry_exception_types=(ValueError,),
        max_attempt_number=max_attempts,
        wait_exponential_jitter=False,
    )


def test_batch_all_succeed() -> None:
    """All items succeed on first attempt."""
    runnable = _make_flaky_runnable(fail_always=set())
    results = runnable.batch([0, 1, 2], return_exceptions=True)
    assert results == [0, 10, 20]


def test_batch_all_fail() -> None:
    """All items always fail."""
    runnable = _make_flaky_runnable(fail_always={0, 1, 2}, max_attempts=2)
    results = runnable.batch([0, 1, 2], return_exceptions=True)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, Exception)


def test_batch_partial_failure_output_alignment() -> None:
    """Items that succeed on retry and items that always fail stay at correct indices.

    This is the core bug from issue #35475: when the last retry attempt
    has mixed results (some succeed, some fail), pop(0) misaligns outputs.
    """
    # Input 0: fails first attempt, succeeds on retry
    # Input 1: always fails
    # Input 2: succeeds immediately
    runnable = _make_flaky_runnable(
        fail_always={1},
        fail_first_n={0: 1},
        max_attempts=3,
    )
    results = runnable.batch([0, 1, 2], return_exceptions=True)

    assert results[0] == 0, f"Expected 0, got {results[0]}"
    assert isinstance(results[1], Exception), (
        f"Expected exception at idx 1, got {results[1]}"
    )
    assert results[2] == 20, f"Expected 20, got {results[2]}"


def test_batch_mixed_last_attempt_corruption() -> None:
    """Specifically target the pop(0) bug.

    Last attempt returns [success, failure, success]. Without the fix,
    pop(0) would assign the success to the wrong index.
    """
    # All fail first attempt; on second attempt, 0 and 2 succeed, 1 fails
    runnable = _make_flaky_runnable(
        fail_always={1},
        fail_first_n={0: 1, 2: 1},
        max_attempts=3,
    )
    results = runnable.batch([0, 1, 2], return_exceptions=True)

    assert results[0] == 0, f"Expected 0, got {results[0]}"
    assert isinstance(results[1], Exception), (
        f"Expected exception at idx 1, got {results[1]}"
    )
    assert results[2] == 20, f"Expected 20, got {results[2]}"


def test_batch_staggered_retries() -> None:
    """Items succeed at different retry rounds.

    Verifies index tracking across multiple attempts.
    """
    # Input 0: succeeds immediately
    # Input 1: fails once, succeeds on 2nd attempt
    # Input 2: fails twice, succeeds on 3rd attempt
    # Input 3: always fails
    runnable = _make_flaky_runnable(
        fail_always={3},
        fail_first_n={1: 1, 2: 2},
        max_attempts=3,
    )
    results = runnable.batch([0, 1, 2, 3], return_exceptions=True)

    assert results[0] == 0
    assert results[1] == 10
    assert results[2] == 20
    assert isinstance(results[3], Exception)


@pytest.mark.asyncio
async def test_abatch_partial_failure_output_alignment() -> None:
    """Test that abatch outputs are aligned to their original input indices.

    Must be at correct indices.
    """
    runnable = _make_flaky_runnable(
        fail_always={1},
        fail_first_n={0: 1},
        max_attempts=3,
    )
    results = await runnable.abatch([0, 1, 2], return_exceptions=True)

    assert results[0] == 0, f"Expected 0, got {results[0]}"
    assert isinstance(results[1], Exception), (
        f"Expected exception at idx 1, got {results[1]}"
    )
    assert results[2] == 20, f"Expected 20, got {results[2]}"


@pytest.mark.asyncio
async def test_abatch_mixed_last_attempt_corruption() -> None:
    """Async version: specifically targets the pop(0) bug."""
    runnable = _make_flaky_runnable(
        fail_always={1},
        fail_first_n={0: 1, 2: 1},
        max_attempts=3,
    )
    results = await runnable.abatch([0, 1, 2], return_exceptions=True)

    assert results[0] == 0, f"Expected 0, got {results[0]}"
    assert isinstance(results[1], Exception), (
        f"Expected exception at idx 1, got {results[1]}"
    )
    assert results[2] == 20, f"Expected 20, got {results[2]}"
