import os
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator

from langchain.document_loaders.blob_loaders.schema import Blob


@contextmanager
def temp_file(content: bytes) -> Generator[Path, None, None]:
    """Yield a temporary field with some content."""
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(content)
        path = Path(temp_file.name)
    try:
        yield path
    finally:
        os.remove(str(path))


def test_blob_as_string() -> None:
    """Test reading blob IO if blob content hasn't been read yet."""
    data = b"Hello, World!"
    blob = Blob(data=BytesIO(data))
    assert blob.as_string() == "Hello, World!"


def test_blob_from_file_binary() -> None:
    """Test reading blob from a file path."""
    content = b"Hello, World!"

    with temp_file(content) as temp_path:
        blob = Blob.from_path(temp_path)
        assert blob.data == content
        assert blob.mimetype is None
        assert blob.encoding is None
        assert blob.path_like == str(temp_path)


def test_blob_from_file_binary_with_str_path() -> None:
    """Test reading blob from a file path."""
    content = b"Hello, World!"

    with temp_file(content) as temp_path:
        blob = Blob.from_path(str(temp_path))
        assert blob.data == content
        assert blob.mimetype is None
        assert blob.encoding is None
        assert blob.path_like == str(temp_path)
