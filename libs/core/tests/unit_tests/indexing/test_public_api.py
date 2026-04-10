from langchain_core.indexing import __all__
import pytest
from typing import List, AsyncIterable

from langchain_core.indexing.api import _batch, _abatch


def test_all() -> None:
    """Use to catch obvious breaking changes."""
    assert list(__all__) == sorted(__all__, key=str)
    assert set(__all__) == {
        "aindex",
        "DeleteResponse",
        "DocumentIndex",
        "index",
        "IndexingResult",
        "InMemoryRecordManager",
        "RecordManager",
        "UpsertResponse",
    }

async def async_iter_from_list(data: List):
    for item in data:
        yield item


def test_batch_size_zero_raises_error() -> None:
    """Test that _batch raises ValueError when size <= 0."""
    with pytest.raises(ValueError, match="Batch size must be a positive integer"):
        list(_batch(0, [1, 2, 3]))

    with pytest.raises(ValueError, match="Batch size must be a positive integer"):
        list(_batch(-1, [1, 2, 3]))


def test_batch_normal_behavior() -> None:
    """Test that _batch works correctly for valid sizes."""
    assert list(_batch(2, [1, 2, 3, 4, 5])) == [[1, 2], [3, 4], [5]]
    assert list(_batch(10, [1, 2, 3])) == [[1, 2, 3]]
    assert list(_batch(2, [])) == []


@pytest.mark.asyncio
async def test_abatch_size_zero_raises_error() -> None:
    """Test that _abatch raises ValueError when size <= 0."""
    with pytest.raises(ValueError, match="Batch size must be a positive integer"):
        async for _ in _abatch(0, async_iter_from_list([1, 2, 3])):
            pass

    with pytest.raises(ValueError, match="Batch size must be a positive integer"):
        async for _ in _abatch(-5, async_iter_from_list([1, 2, 3])):
            pass


@pytest.mark.asyncio
async def test_abatch_normal_behavior() -> None:
    """Test that _abatch works correctly for valid sizes."""
    data = [1, 2, 3, 4, 5]
    batches = []
    async for b in _abatch(2, async_iter_from_list(data)):
        batches.append(b)
    assert batches == [[1, 2], [3, 4], [5]]

    batches = []
    async for b in _abatch(10, async_iter_from_list(data)):
        batches.append(b)
    assert batches == [[1, 2, 3, 4, 5]]

    batches = []
    async for b in _abatch(2, async_iter_from_list([])):
        batches.append(b)
    assert batches == []