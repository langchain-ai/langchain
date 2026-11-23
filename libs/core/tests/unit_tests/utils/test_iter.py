import pytest

from langchain_core.utils.iter import batch_iterate


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
