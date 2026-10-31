import inspect
from collections.abc import Generator, Iterator

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


def test_tee_context_exit_zero_started() -> None:
    closed = []

    class Source:
        def __iter__(self) -> "Source":
            return self

        def __next__(self) -> int:
            return 1

        def close(self) -> None:
            closed.append(True)

    def gen_source() -> Generator[int, None, None]:
        yield 1

    stream = Source()
    with Tee(stream) as peers:
        assert not closed
        assert len(peers) == 2
    assert closed == [True]

    gen_stream = gen_source()
    with Tee(gen_stream):
        assert inspect.getgeneratorstate(gen_stream) == "GEN_CREATED"
    assert inspect.getgeneratorstate(gen_stream) == "GEN_CLOSED"


def test_tee_context_exit_one_started() -> None:
    closed = []

    def source() -> Iterator[int]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = source()
    with Tee(stream) as peers:
        assert not closed
        assert next(peers[0]) == 1
        assert not closed
    assert closed == [True]


def test_tee_context_exit_two_started() -> None:
    closed = []

    def source() -> Iterator[int]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = source()
    with Tee(stream) as peers:
        assert not closed
        assert next(peers[0]) == 1
        assert next(peers[1]) == 1
        assert not closed
    assert closed == [True]


def test_tee_close_with_unstarted_child() -> None:
    closed = []

    def source() -> Iterator[int]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = source()
    tee = Tee(stream)
    assert not closed
    assert next(tee[0]) == 1
    assert not closed
    tee.close()
    assert closed == [True]
    tee.close()
    assert closed == [True]


def test_tee_repeated_close() -> None:
    closed_count = 0

    class Source:
        def __iter__(self) -> "Source":
            return self

        def __next__(self) -> int:
            return 1

        def close(self) -> None:
            nonlocal closed_count
            closed_count += 1

    source = Source()
    tee = Tee(source)
    assert next(tee[0]) == 1
    assert next(tee[1]) == 1
    assert closed_count == 0
    tee.close()
    assert closed_count == 1
    tee.close()
    assert closed_count == 1


def test_tee_mixed_usage_three_children() -> None:
    closed = []

    def source() -> Iterator[int]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = source()
    tee = Tee(stream, n=3)
    assert not closed
    assert next(tee[0]) == 1
    # child 1 is never started
    assert next(tee[2]) == 1
    assert not closed
    tee.close()
    assert len(tee._buffers) == 0
    assert closed == [True]
