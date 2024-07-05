from typing import AsyncIterator, List

import pytest

from langchain_core.utils.aiter import abatch_iterate


@pytest.mark.parametrize(
    "input_size, input_iterable, expected_output",
    [
        (2, [1, 2, 3, 4, 5], [[1, 2], [3, 4], [5]]),
        (3, [10, 20, 30, 40, 50], [[10, 20, 30], [40, 50]]),
        (1, [100, 200, 300], [[100], [200], [300]]),
        (4, [], []),
    ],
)
async def test_abatch_iterate(
    input_size: int, input_iterable: List[str], expected_output: List[str]
) -> None:
    """Test batching function."""

    async def _to_async_iterable(iterable: List[str]) -> AsyncIterator[str]:
        for item in iterable:
            yield item

    iterator_ = abatch_iterate(input_size, _to_async_iterable(input_iterable))

    assert isinstance(iterator_, AsyncIterator)

    output = [el async for el in iterator_]
    assert output == expected_output
