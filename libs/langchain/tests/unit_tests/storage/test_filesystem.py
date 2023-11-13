import tempfile
from typing import Generator

import pytest

from langchain.storage.exceptions import InvalidKeyException
from langchain.storage.file_system import LocalFileStore


@pytest.fixture
def file_store() -> Generator[LocalFileStore, None, None]:
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Instantiate the LocalFileStore with the temporary directory as the root path
        store = LocalFileStore(temp_dir)
        yield store


def test_mset_and_mget(file_store: LocalFileStore) -> None:
    # Set values for keys
    key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
    file_store.mset(key_value_pairs)

    # Get values for keys
    values = file_store.mget(["key1", "key2"])

    # Assert that the retrieved values match the original values
    assert values == [b"value1", b"value2"]


def test_mdelete(file_store: LocalFileStore) -> None:
    # Set values for keys
    key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
    file_store.mset(key_value_pairs)

    # Delete keys
    file_store.mdelete(["key1"])

    # Check if the deleted key is present
    values = file_store.mget(["key1"])

    # Assert that the value is None after deletion
    assert values == [None]


def test_set_invalid_key(file_store: LocalFileStore) -> None:
    """Test that an exception is raised when an invalid key is set."""
    # Set a key-value pair
    key = "crying-cat/😿"
    value = b"This is a test value"
    with pytest.raises(InvalidKeyException):
        file_store.mset([(key, value)])


def test_set_key_and_verify_content(file_store: LocalFileStore) -> None:
    """Test that the content of the file is the same as the value set."""
    # Set a key-value pair
    key = "test_key"
    value = b"This is a test value"
    file_store.mset([(key, value)])

    # Verify the content of the actual file
    full_path = file_store._get_full_path(key)
    assert full_path.exists()
    assert full_path.read_bytes() == b"This is a test value"


def test_yield_keys(file_store: LocalFileStore) -> None:
    # Set values for keys
    key_value_pairs = [("key1", b"value1"), ("subdir/key2", b"value2")]
    file_store.mset(key_value_pairs)

    # Iterate over keys
    keys = list(file_store.yield_keys())

    # Assert that the yielded keys match the expected keys
    expected_keys = ["key1", "subdir/key2"]
    assert keys == expected_keys
