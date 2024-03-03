from typing import Any, List
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders import BatchLoader
from langchain_community.document_loaders.base import BaseLoader


# Mocking BaseLoader and its load method
class MockLoader(BaseLoader):
    """MockLoader class used to test BatchLoader"""

    def __init__(self, **kwargs: Any):
        # Mocking initialization to allow for any kwargs
        pass

    def load(self) -> List[Document]:
        return [Document(page_content="", metadata={"doc_id": i}) for i in range(2)]


@pytest.fixture(name="batch_loader")
def fixture_batch_loader() -> BatchLoader:
    """Fixture to create a BatchLoader instance with 2 loader_args so that
    load() is called twice in the batch loader.

    Returns:
        BatchLoader: BatchLoader instance
    """
    return BatchLoader(MockLoader, {"arg1": ["val1", "val2"]}, show_progress=False)


def test_loader_args_list_length_not_equal() -> None:
    """Test that an error is raised if the length of the values of the
    loader_args dict are not equal.
    """
    with pytest.raises(ValueError):
        BatchLoader(
            MockLoader,
            {"arg1": ["val1", "val2"], "arg2": ["val1"]},
            show_progress=False,
        )


def test_missing_tqdm() -> None:
    """Test that an error is raised if tqdm is not installed"""
    # Remove tqdm from sys.modules to simulate a missing import
    with patch.dict("sys.modules", {"tqdm": None}):
        with pytest.raises(ImportError):
            BatchLoader(
                MockLoader,
                {"file_path": ["path1", "path2"]},
                show_progress=True,
            ).load()


def test_load_invalid_method(batch_loader: BatchLoader) -> None:
    """Test that an error is raised if the method is invalid"""
    with pytest.raises(ValueError):
        batch_loader.method = "invalid"  # type: ignore
        batch_loader.load()


def test_use_load_with_async_method(batch_loader: BatchLoader) -> None:
    """Test that an error is raised if the method is invalid"""
    with pytest.raises(ValueError):
        batch_loader.method = "async"
        batch_loader.load()


def test_load_sequential(batch_loader: BatchLoader) -> None:
    """Test that load() calls the load() method of the loader sequentially"""
    documents = batch_loader.load()
    assert len(documents) == 4
    # Assert documents have been loaded in the right order
    assert [doc.metadata["doc_id"] for doc in documents] == [0, 1, 0, 1]


def test_load_thread(batch_loader: BatchLoader) -> None:
    """Test that load() calls the load() method of the loader in threads"""
    batch_loader.method = "thread"
    batch_loader.max_workers = 2
    documents = batch_loader.load()
    assert len(documents) == 4
    assert [doc.metadata["doc_id"] for doc in documents] == [0, 1, 0, 1]


def test_load_process(batch_loader: BatchLoader) -> None:
    """Test that load() calls the load() method of the loader in processes"""
    batch_loader.method = "process"
    batch_loader.max_workers = 2
    documents = batch_loader.load()
    assert len(documents) == 4
    assert [doc.metadata["doc_id"] for doc in documents] == [0, 1, 0, 1]


@pytest.mark.asyncio
async def test_load_async(batch_loader: BatchLoader) -> None:
    """Test that load() calls the load() method of the loader in processes"""
    batch_loader.method = "async"
    documents = await batch_loader.aload()
    assert len(documents) == 4
    assert [doc.metadata["doc_id"] for doc in documents] == [0, 1, 0, 1]
