import tempfile
from typing import Generator

import pytest

from langchain.docstore._kw_storage import create_kw_docstore
from langchain.load.dump import dumpd
from langchain.schema import Document
from langchain.storage.file_system import LocalFileStore


@pytest.fixture
def file_store() -> Generator[LocalFileStore, None, None]:
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Instantiate the LocalFileStore with the temporary directory as the root path
        store = LocalFileStore(temp_dir)
        yield store


def test_create_docstore(file_store: LocalFileStore) -> None:
    """Test that a docstore is created from a base store."""
    docstore = create_kw_docstore(file_store)
    docstore.mset([("key1", Document(page_content="hello", metadata={"key": "value"}))])
    fetched_doc = docstore.mget(["key1"])
    assert fetched_doc[0].page_content == "hello"
    assert dumpd(fetched_doc) == ""
