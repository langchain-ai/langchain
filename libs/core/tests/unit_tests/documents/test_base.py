"""Tests for `langchain_core.documents.base.Blob`."""

from langchain_core.documents.base import Blob


def test_as_bytes_io_with_string_data() -> None:
    blob = Blob.from_data("hello")
    with blob.as_bytes_io() as stream:
        assert stream.read() == b"hello"


def test_as_bytes_io_with_string_data_respects_encoding() -> None:
    blob = Blob.from_data("café", encoding="utf-16")
    with blob.as_bytes_io() as stream:
        assert stream.read() == "café".encode("utf-16")


def test_as_bytes_io_with_bytes_data() -> None:
    blob = Blob.from_data(b"\x00\x01\x02")
    with blob.as_bytes_io() as stream:
        assert stream.read() == b"\x00\x01\x02"
