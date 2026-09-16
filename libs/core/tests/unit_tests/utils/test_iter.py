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


def test_tee_class_basic_operations() -> None:
    """Test Tee class indexing, slicing, iteration, and context manager."""
    from langchain_core.utils.iter import NoLock, Tee, safetee

    # Test safetee alias
    assert safetee is Tee

    # Test NoLock dummy context manager
    lock = NoLock()
    with lock:
        pass

    # Test Tee splitting
    source = [1, 2, 3, 4]
    tee_obj = Tee(source, n=3)
    assert len(tee_obj) == 3

    # Test slicing and indexing
    first_child = tee_obj[0]
    sliced_children = tee_obj[1:3]
    assert len(sliced_children) == 2

    # Verify each child consumes items independently
    assert list(first_child) == source
    assert list(sliced_children[0]) == source

    # Test context manager enter/exit
    with Tee([10, 20], n=2) as t:
        assert len(t) == 2
        assert list(t[0]) == [10, 20]
