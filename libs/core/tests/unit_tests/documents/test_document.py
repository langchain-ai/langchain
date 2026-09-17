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


def test_blob_as_string_strips_utf8_bom(tmp_path: Path) -> None:
    """Test that Blob.as_string strips UTF-8 BOM (#40441)."""
    bom_file = tmp_path / "bom_doc.md"
    bom_file.write_text("# Heading\n\nbody\n", encoding="utf-8-sig")

    blob_from_path = Blob.from_path(bom_file)
    assert blob_from_path.as_string() == "# Heading\n\nbody\n"

    blob_from_bytes = Blob.from_data(b"\xef\xbb\xbf# Heading\n\nbody\n")
    assert blob_from_bytes.as_string() == "# Heading\n\nbody\n"

