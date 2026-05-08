"""Tests for `langchain_core.documents.base`."""

from io import BufferedReader, BytesIO
from pathlib import Path

import pytest

from langchain_core.documents.base import Blob


def test_blob_as_bytes_io_with_string_data() -> None:
    """`as_bytes_io` should yield a byte stream when the blob holds string data.

    Regression test for an inconsistency where `as_string` and `as_bytes`
    accepted string data but `as_bytes_io` raised `NotImplementedError`.
    """
    blob = Blob.from_data("hello")
    with blob.as_bytes_io() as stream:
        assert isinstance(stream, BytesIO)
        assert stream.read() == b"hello"


def test_blob_as_bytes_io_with_bytes_data() -> None:
    """`as_bytes_io` should yield a byte stream for raw bytes data."""
    blob = Blob.from_data(b"hello")
    with blob.as_bytes_io() as stream:
        assert isinstance(stream, BytesIO)
        assert stream.read() == b"hello"


def test_blob_as_bytes_io_respects_encoding() -> None:
    """String data should be encoded with the blob's `encoding`."""
    blob = Blob.from_data("héllo", encoding="latin-1")
    with blob.as_bytes_io() as stream:
        assert stream.read() == "héllo".encode("latin-1")


def test_blob_as_bytes_io_consistent_with_as_bytes_for_string() -> None:
    """`as_bytes_io` and `as_bytes` should produce the same bytes for string data."""
    blob = Blob.from_data("hello world")
    with blob.as_bytes_io() as stream:
        from_stream = stream.read()
    assert from_stream == blob.as_bytes()


def test_blob_as_bytes_io_with_path(tmp_path: Path) -> None:
    """`as_bytes_io` should yield a buffered file reader when reading from a path."""
    file_path = tmp_path / "blob.txt"
    file_path.write_bytes(b"hello from disk")
    blob = Blob.from_path(file_path)
    with blob.as_bytes_io() as stream:
        assert isinstance(stream, BufferedReader)
        assert stream.read() == b"hello from disk"


def test_blob_as_bytes_io_raises_when_no_data() -> None:
    """`as_bytes_io` should raise `NotImplementedError` when the blob has no data."""
    blob = Blob(data=None, path=None)
    with pytest.raises(NotImplementedError), blob.as_bytes_io():
        pass
