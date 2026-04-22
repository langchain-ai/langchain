from langchain_core.documents.base import Blob


def test_as_bytes_io_from_string_data() -> None:
    blob = Blob.from_data("hello")
    with blob.as_bytes_io() as f:
        assert f.read() == b"hello"


def test_as_bytes_io_from_bytes_data() -> None:
    blob = Blob.from_data(b"hello")
    with blob.as_bytes_io() as f:
        assert f.read() == b"hello"


def test_as_bytes_io_respects_encoding() -> None:
    blob = Blob.from_data("héllo", encoding="latin-1")
    with blob.as_bytes_io() as f:
        assert f.read() == "héllo".encode("latin-1")
