from pathlib import Path

import pytest

from langchain_core.documents.base import Blob


def test_blob_from_data() -> None:
    blob = Blob.from_data("hello", mime_type="text/plain", metadata={"a": 1})
    assert blob.data == "hello"
    assert blob.mimetype == "text/plain"
    assert blob.metadata == {"a": 1}
    assert blob.as_string() == "hello"
    assert blob.as_bytes() == b"hello"


def test_blob_from_path(tmp_path: Path) -> None:
    p = tmp_path / "test.txt"
    p.write_text("hello world", encoding="utf-8")

    # Test without explicit mime_type, let it guess
    blob = Blob.from_path(str(p))
    assert blob.path == str(p)
    assert blob.mimetype == "text/plain"
    assert blob.as_string() == "hello world"

    # Test with explicit mime_type
    blob2 = Blob.from_path(str(p), mime_type="application/text")
    assert blob2.mimetype == "application/text"


def test_blob_invalid_init() -> None:
    with pytest.raises(ValueError, match="Either data or path must be provided"):
        Blob(mimetype="text/plain")


def test_blob_as_bytes_io() -> None:
    blob = Blob.from_data(b"binary data")
    with blob.as_bytes_io() as f:
        assert f.read() == b"binary data"
