from pathlib import Path

import pytest

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


def test_blob_rejects_none_data_and_path() -> None:
    with pytest.raises(ValueError, match="Either data or path must be provided"):
        Blob(data=None)


def test_metadata_allows_non_string_keys(tmp_path: Path) -> None:
    metadata = {1: "one"}

    doc = Document(page_content="foo", metadata=metadata)
    blob_from_data = Blob.from_data("foo", metadata=metadata)
    blob_from_path = Blob.from_path(tmp_path / "foo.txt", metadata=metadata)

    assert doc.metadata == metadata
    assert blob_from_data.metadata == metadata
    assert blob_from_path.metadata == metadata
