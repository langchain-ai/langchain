import codecs
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


def test_metadata_allows_non_string_keys(tmp_path: Path) -> None:
    metadata = {1: "one"}

    doc = Document(page_content="foo", metadata=metadata)
    blob_from_data = Blob.from_data("foo", metadata=metadata)
    blob_from_path = Blob.from_path(tmp_path / "foo.txt", metadata=metadata)

    assert doc.metadata == metadata
    assert blob_from_data.metadata == metadata
    assert blob_from_path.metadata == metadata


CONTENT = "# Heading\n\ncafé body\n"


@pytest.mark.parametrize("suffix", [".txt", ".md", ".csv"])
def test_blob_from_path_strips_utf8_bom(tmp_path: Path, suffix: str) -> None:
    """A file saved with a UTF-8 BOM should not leak the BOM into the decoded string.

    Notepad, Excel's "CSV UTF-8" export and PowerShell redirection all write a BOM by
    default, so a file that has been opened and re-saved on Windows arrives with one.
    Decoded as plain `utf-8` the BOM survives as a zero-width `U+FEFF` at the start of
    the content, from where it flows into splitters, embeddings and prompts.
    """
    path = tmp_path / f"bom{suffix}"
    path.write_text(CONTENT, encoding="utf-8-sig")
    assert path.read_bytes().startswith(codecs.BOM_UTF8)

    blob = Blob.from_path(path)

    assert blob.as_string() == CONTENT
    assert not blob.as_string().startswith("﻿")


def test_blob_from_data_strips_utf8_bom() -> None:
    """The bytes path has the same guarantee as the file path."""
    blob = Blob.from_data(codecs.BOM_UTF8 + CONTENT.encode("utf-8"))

    assert blob.as_string() == CONTENT


@pytest.mark.parametrize("suffix", [".txt", ".md", ".csv"])
def test_blob_from_path_without_bom_is_unchanged(tmp_path: Path, suffix: str) -> None:
    """Files without a BOM must decode exactly as before.

    `utf-8-sig` decodes plain UTF-8 identically to `utf-8`, so this is the
    backward-compatibility guarantee, asserted rather than assumed.
    """
    path = tmp_path / f"plain{suffix}"
    path.write_text(CONTENT, encoding="utf-8")
    assert not path.read_bytes().startswith(codecs.BOM_UTF8)

    assert Blob.from_path(path).as_string() == CONTENT


def test_blob_explicit_encoding_still_honoured(tmp_path: Path) -> None:
    """An explicit `encoding` argument keeps its exact meaning."""
    path = tmp_path / "bom.txt"
    path.write_text(CONTENT, encoding="utf-8-sig")

    # Asking for plain utf-8 explicitly preserves the BOM, as it always did.
    assert Blob.from_path(path, encoding="utf-8").as_string().startswith("﻿")
    # A non-UTF-8 codec is untouched by this change.
    latin = tmp_path / "latin.txt"
    latin.write_bytes("café".encode("latin-1"))
    assert Blob.from_path(latin, encoding="latin-1").as_string() == "café"
