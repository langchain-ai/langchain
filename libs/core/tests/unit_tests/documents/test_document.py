from pathlib import Path

from langchain_core.documents import Document
from langchain_core.documents.base import Blob


def test_init() -> None:
    for doc in [
        Document(page_content="foo"),
        Document(page_content="foo", metadata={"a": 1}),
        Document(page_content="foo", id=None),
        Document(page_content="foo", id="1"),
        Document(page_content="foo", id=1),
    ]:
        assert isinstance(doc, Document)


def test_metadata_allows_non_string_keys(tmp_path: Path) -> None:
    metadata = {1: "one"}

    doc = Document(page_content="foo", metadata=metadata)
    blob_from_data = Blob.from_data("foo", metadata=metadata)
    blob_from_path = Blob.from_path(tmp_path / "foo.txt", metadata=metadata)

    assert doc.metadata == metadata
    assert blob_from_data.metadata == metadata
    assert blob_from_path.metadata == metadata


def test_blob_as_string_with_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_text("Hello, world!", encoding="utf-8-sig")

    blob = Blob.from_path(path)

    result = blob.as_string()

    assert result == "Hello, world!"


def test_blob_from_data_with_utf8_bom() -> None:
    data = b"\xef\xbb\xbfHello, world!"

    blob = Blob.from_data(data)

    result = blob.as_string()

    assert result == "Hello, world!"
