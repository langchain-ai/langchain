from collections.abc import AsyncIterator

import pytest

from langchain_core.utils.aiter import abatch_iterate


@pytest.mark.parametrize(
    ("input_size", "input_iterable", "expected_output"),
    [
        (2, [1, 2, 3, 4, 5], [[1, 2], [3, 4], [5]]),
        (3, [10, 20, 30, 40, 50], [[10, 20, 30], [40, 50]]),
        (1, [100, 200, 300], [[100], [200], [300]]),
        (4, [], []),
    ],
)
async def test_abatch_iterate(
    input_size: int, input_iterable: list[str], expected_output: list[list[str]]
) -> None:
    """Test batching function."""

    async def _to_async_iterable(iterable: list[str]) -> AsyncIterator[str]:
        for item in iterable:
            yield item

    iterator_ = abatch_iterate(input_size, _to_async_iterable(input_iterable))

    assert isinstance(iterator_, AsyncIterator)

    output = [el async for el in iterator_]
    assert output == expected_output


@pytest.mark.parametrize("bad_size", [0, -1, -100])
async def test_abatch_iterate_invalid_size_raises(bad_size: int) -> None:
    """abatch_iterate must raise ValueError for non-positive size."""

    async def _gen() -> AsyncIterator[int]:
        yield 1

    with pytest.raises(ValueError, match="positive integer"):
        async for _ in abatch_iterate(bad_size, _gen()):
            pass


async def test_abatch_iterate_accepts_async_generator() -> None:
    """abatch_iterate must correctly batch a real async generator."""

    async def _gen() -> AsyncIterator[int]:
        for i in range(5):
            yield i

    result = [batch async for batch in abatch_iterate(2, _gen())]
    assert result == [[0, 1], [2, 3], [4]]

