import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator

from langchain.document_loaders.blob_loaders.schema import Blob, BlobLoader


@contextmanager
def get_temp_file(content: bytes) -> Generator[Path, None, None]:
    """Yield a temporary field with some content."""
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(content)
        path = Path(temp_file.name)
    try:
        yield path
    finally:
        os.remove(str(path))


def test_blob_initialized_with_binary_data() -> None:
    """Test reading blob IO if blob content hasn't been read yet."""
    data = b"Hello, World!"
    blob = Blob(data=data)
    assert blob.as_string() == "Hello, World!"
    assert blob.as_bytes() == data
    assert blob.source is None
    with blob.as_bytes_io() as bytes_io:
        assert bytes_io.read() == data


def test_blob_from_pure_path() -> None:
    """Test reading blob from a file path."""
    content = b"Hello, World!"

    with get_temp_file(content) as temp_path:
        assert isinstance(temp_path, Path)
        blob = Blob.from_path(temp_path)
        assert blob.encoding == "utf-8"  # Default encoding
        assert blob.path == temp_path
        assert blob.source == str(temp_path)
        assert blob.data is None
        assert blob.as_bytes() == content
        assert blob.as_string() == "Hello, World!"
        with blob.as_bytes_io() as bytes_io:
            assert bytes_io.read() == content


def test_blob_from_str_path() -> None:
    """Test reading blob from a file path."""
    content = b"Hello, World!"

    with get_temp_file(content) as temp_path:
        str_path = str(temp_path)
        assert isinstance(str_path, str)
        blob = Blob.from_path(str_path)
        assert blob.encoding == "utf-8"  # Default encoding
        assert blob.path == str(temp_path)
        assert blob.source == str(temp_path)
        assert blob.data is None
        assert blob.as_bytes() == content
        assert blob.as_string() == "Hello, World!"
        with blob.as_bytes_io() as bytes_io:
            assert bytes_io.read() == content


def test_blob_loader() -> None:
    """Simple test that verifies that we can implement a blob loader."""

    class TestLoader(BlobLoader):
        def yield_blobs(self) -> Generator[Blob, None, None]:
            yield Blob(data=b"Hello, World!")

    assert list(TestLoader().yield_blobs()) == [Blob(data=b"Hello, World!")]
