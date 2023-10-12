import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator, Iterable, Optional

import pytest

from langchain.document_loaders.blob_loaders.schema import Blob, BlobLoader, PathLike


@contextmanager
def get_temp_file(
    content: bytes, suffix: Optional[str] = None
) -> Generator[Path, None, None]:
    """Yield a temporary field with some content."""
    with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
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

    with get_temp_file(content, suffix=".html") as temp_path:
        assert isinstance(temp_path, Path)
        blob = Blob.from_path(temp_path)
        assert blob.encoding == "utf-8"  # Default encoding
        assert blob.path == temp_path
        assert blob.mimetype == "text/html"
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


def test_blob_from_str_data() -> None:
    """Test reading blob from a file path."""
    content = b"Hello, World!"
    blob = Blob.from_data(content)
    assert blob.encoding == "utf-8"  # Default encoding
    assert blob.path is None
    assert blob.mimetype is None
    assert blob.source is None
    assert blob.data == b"Hello, World!"
    assert blob.as_bytes() == content
    assert blob.as_string() == "Hello, World!"
    with blob.as_bytes_io() as bytes_io:
        assert bytes_io.read() == content


def test_blob_mimetype_from_str_data() -> None:
    """Test reading blob from a file path."""
    content = b"Hello, World!"
    mimetype = "text/html"
    blob = Blob.from_data(content, mime_type=mimetype)
    assert blob.mimetype == mimetype


@pytest.mark.parametrize(
    "path, mime_type, guess_type, expected_mime_type",
    [
        ("test.txt", None, True, "text/plain"),
        ("test.txt", None, False, None),
        ("test.html", None, True, "text/html"),
        ("test.html", None, False, None),
        ("test.html", "user_forced_value", True, "user_forced_value"),
        (Path("test.html"), "user_forced_value", True, "user_forced_value"),
        (Path("test.html"), None, True, "text/html"),
    ],
)
def test_mime_type_inference(
    path: PathLike, mime_type: str, guess_type: bool, expected_mime_type: Optional[str]
) -> None:
    """Tests mimetype inference based on options and path."""
    blob = Blob.from_path(path, mime_type=mime_type, guess_type=guess_type)
    assert blob.mimetype == expected_mime_type


def test_blob_initialization_validator() -> None:
    """Test that blob initialization validates the arguments."""
    with pytest.raises(ValueError, match="Either data or path must be provided"):
        Blob()

    assert Blob(data=b"Hello, World!") is not None
    assert Blob(path="some_path") is not None


def test_blob_loader() -> None:
    """Simple test that verifies that we can implement a blob loader."""

    class TestLoader(BlobLoader):
        def yield_blobs(self) -> Iterable[Blob]:
            """Yield blob implementation."""
            yield Blob(data=b"Hello, World!")

    assert list(TestLoader().yield_blobs()) == [Blob(data=b"Hello, World!")]
