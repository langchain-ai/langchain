from typing import List

import pytest

from langchain.utils.iter import batch_iterate


@pytest.mark.parametrize(
    "input_size, input_iterable, expected_output",
    [
        (2, [1, 2, 3, 4, 5], [[1, 2], [3, 4], [5]]),
        (3, [10, 20, 30, 40, 50], [[10, 20, 30], [40, 50]]),
        (1, [100, 200, 300], [[100], [200], [300]]),
        (4, [], []),
    ],
)
def test_batch_iterate(
    input_size: int, input_iterable: List[str], expected_output: List[str]
) -> None:
    """Test batching function."""
    assert list(batch_iterate(input_size, input_iterable)) == expected_output
