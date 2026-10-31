import inspect
from collections.abc import AsyncGenerator, AsyncIterator

import pytest

from langchain_core.utils.aiter import Tee, abatch_iterate


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
async def test_abatch_iterate(
    input_size: int | None,
    input_iterable: list[str],
    expected_output: list[list[str]],
) -> None:
    """Test batching function."""

    async def _to_async_iterable(iterable: list[str]) -> AsyncIterator[str]:
        for item in iterable:
            yield item

    iterator_ = abatch_iterate(input_size, _to_async_iterable(input_iterable))

    assert isinstance(iterator_, AsyncIterator)

    output = [el async for el in iterator_]
    assert output == expected_output


@pytest.mark.parametrize("input_size", [0, -1])
async def test_abatch_iterate_invalid_size(input_size: int) -> None:
    """Non-positive sizes should raise instead of silently discarding data."""

    async def _to_async_iterable(iterable: list[int]) -> AsyncIterator[int]:
        for item in iterable:
            yield item

    with pytest.raises(ValueError, match="positive integer"):
        _ = [el async for el in abatch_iterate(input_size, _to_async_iterable([1, 2]))]


async def test_atee_context_exit_zero_started() -> None:
    closed = []

    class Source:
        def __aiter__(self) -> "Source":
            return self

        async def __anext__(self) -> int:
            return 1

        async def aclose(self) -> None:
            closed.append(True)

    async def gen_source() -> AsyncGenerator[int, None]:
        yield 1

    stream = Source()
    async with Tee(stream) as peers:
        assert not closed
        assert len(peers) == 2
    assert closed == [True]

    gen_stream = gen_source()
    async with Tee(gen_stream):
        assert inspect.getasyncgenstate(gen_stream) == "AGEN_CREATED"
    assert inspect.getasyncgenstate(gen_stream) == "AGEN_CLOSED"


async def test_atee_context_exit_one_started() -> None:
    closed = []

    async def source() -> AsyncIterator[int]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = source()
    async with Tee(stream) as peers:
        assert not closed
        assert await anext(peers[0]) == 1
        assert not closed
    assert closed == [True]


async def test_atee_context_exit_two_started() -> None:
    closed = []

    async def source() -> AsyncIterator[int]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = source()
    async with Tee(stream) as peers:
        assert not closed
        assert await anext(peers[0]) == 1
        assert await anext(peers[1]) == 1
        assert not closed
    assert closed == [True]


async def test_atee_aclose_with_unstarted_child() -> None:
    closed = []

    async def source() -> AsyncIterator[int]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = source()
    tee = Tee(stream)
    assert not closed
    assert await anext(tee[0]) == 1
    assert not closed
    await tee.aclose()
    assert closed == [True]
    await tee.aclose()
    assert closed == [True]


async def test_atee_repeated_aclose() -> None:
    closed_count = 0

    class Source:
        def __aiter__(self) -> "Source":
            return self

        async def __anext__(self) -> int:
            return 1

        async def aclose(self) -> None:
            nonlocal closed_count
            closed_count += 1

    source = Source()
    tee = Tee(source)
    assert await anext(tee[0]) == 1
    assert await anext(tee[1]) == 1
    assert closed_count == 0
    await tee.aclose()
    assert closed_count == 1
    await tee.aclose()
    assert closed_count == 1


async def test_atee_mixed_usage_three_children() -> None:
    closed = []

    async def source() -> AsyncIterator[int]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = source()
    tee = Tee(stream, n=3)
    assert not closed
    assert await anext(tee[0]) == 1
    # child 1 is never started
    assert await anext(tee[2]) == 1
    assert not closed
    await tee.aclose()
    assert len(tee._buffers) == 0
    assert closed == [True]
