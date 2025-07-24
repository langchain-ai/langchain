import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from langchain_core.stores import InvalidKeyException

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


@pytest.mark.parametrize(
    ("chmod_dir_s", "chmod_file_s"),
    [("777", "666"), ("770", "660"), ("700", "600")],
)
def test_mset_chmod(chmod_dir_s: str, chmod_file_s: str) -> None:
    chmod_dir = int(chmod_dir_s, base=8)
    chmod_file = int(chmod_file_s, base=8)

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Instantiate the LocalFileStore with a directory inside the temporary directory
        # as the root path
        file_store = LocalFileStore(
            Path(temp_dir) / "store_dir",
            chmod_dir=chmod_dir,
            chmod_file=chmod_file,
        )

        # Set values for keys
        key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
        file_store.mset(key_value_pairs)

        # verify the permissions are set correctly
        # (test only the standard user/group/other bits)
        dir_path = file_store.root_path
        file_path = file_store.root_path / "key1"
        assert (dir_path.stat().st_mode & 0o777) == chmod_dir
        assert (file_path.stat().st_mode & 0o777) == chmod_file


def test_mget_update_atime() -> None:
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Instantiate the LocalFileStore with a directory inside the temporary directory
        # as the root path
        file_store = LocalFileStore(Path(temp_dir) / "store_dir", update_atime=True)

        # Set values for keys
        key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
        file_store.mset(key_value_pairs)

        # Get original access time
        file_path = file_store.root_path / "key1"
        atime1 = file_path.stat().st_atime

        # Get values for keys
        _ = file_store.mget(["key1", "key2"])

        # Make sure the filesystem access time has been updated
        atime2 = file_path.stat().st_atime
        assert atime2 != atime1


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
    expected_keys = ["key1", str(Path("subdir") / "key2")]
    assert keys == expected_keys


def test_catches_forbidden_keys(file_store: LocalFileStore) -> None:
    """Make sure we raise exception on keys that are not allowed; e.g., absolute path"""
    with pytest.raises(InvalidKeyException):
        file_store.mset([("/etc", b"value1")])
    with pytest.raises(InvalidKeyException):
        list(file_store.yield_keys("/etc/passwd"))
    with pytest.raises(InvalidKeyException):
        file_store.mget(["/etc/passwd"])

    # check relative paths
    with pytest.raises(InvalidKeyException):
        list(file_store.yield_keys(".."))

    with pytest.raises(InvalidKeyException):
        file_store.mget(["../etc/passwd"])

    with pytest.raises(InvalidKeyException):
        file_store.mset([("../etc", b"value1")])

    with pytest.raises(InvalidKeyException):
        list(file_store.yield_keys("../etc/passwd"))
