from collections.abc import Generator

import pytest

from langchain_core.utils.iter import Tee, batch_iterate


@pytest.mark.parametrize(
    ("input_size", "input_iterable", "expected_output"),
    [
        (2, [1, 2, 3, 4, 5], [[1, 2], [3, 4], [5]]),
        (3, [10, 20, 30, 40, 50], [[10, 20, 30], [40, 50]]),
        (1, [100, 200, 300], [[100], [200], [300]]),
        (4, [], []),
        (None, [1, 2, 3], [[1, 2, 3]]),
        (None, [], []),
    ],
)
def test_batch_iterate(
    input_size: int | None,
    input_iterable: list[str],
    expected_output: list[list[str]],
) -> None:
    """Test batching function."""
    assert list(batch_iterate(input_size, input_iterable)) == expected_output


@pytest.mark.parametrize("input_size", [0, -1])
def test_batch_iterate_invalid_size(input_size: int) -> None:
    """Non-positive sizes should raise instead of silently discarding data."""
    with pytest.raises(ValueError, match="positive integer"):
        list(batch_iterate(input_size, [1, 2, 3]))


def test_tee_close_closes_source_when_child_never_started() -> None:
    """Closing a Tee must close the source even if a child was never started.

    Regression test for langchain-ai/langchain#40935: closing a never-started
    child generator executes none of its cleanup code, so without an explicit
    source close in `Tee.close` the source generator stays suspended forever.
    """
    closed: list[bool] = []

    def source() -> Generator[int, None, None]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = source()
    try:
        with Tee(stream) as peers:
            assert next(peers[0]) == 1
        assert closed, "source generator was not closed"
    finally:
        stream.close()


def test_tee_close_is_safe_for_source_without_close() -> None:
    """`Tee.close` must not fail when the source has no `close` method."""
    with Tee(iter([1, 2, 3])) as peers:
        assert next(peers[0]) == 1
    # no exception raised
