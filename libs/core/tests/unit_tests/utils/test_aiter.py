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


async def test_atee_aclose_closes_source_when_child_never_started() -> None:
    """Closing a Tee must close the source even if a child was never started.

    Regression test for langchain-ai/langchain#40935: closing a never-started
    child async generator executes none of its cleanup code, so without an
    explicit source close in `Tee.aclose` the source stays suspended forever.
    """
    closed: list[bool] = []

    async def source() -> AsyncGenerator[int, None]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = source()
    try:
        async with Tee(stream) as peers:
            assert await anext(peers[0]) == 1
        assert closed, "source async generator was not closed"
    finally:
        await stream.aclose()


async def test_atee_aclose_is_safe_for_source_without_aclose() -> None:
    """`Tee.aclose` must not fail when the source has no `aclose` method."""

    class _NoCloseAsyncIterator:
        def __init__(self, items: list[int]) -> None:
            self._it = iter(items)

        def __aiter__(self) -> "_NoCloseAsyncIterator":
            return self

        async def __anext__(self) -> int:
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration from None

    async with Tee(_NoCloseAsyncIterator([1, 2, 3])) as peers:
        assert await anext(peers[0]) == 1
    # no exception raised
