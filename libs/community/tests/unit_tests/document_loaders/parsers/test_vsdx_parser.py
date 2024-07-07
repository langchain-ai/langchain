"""Tests for the VSDX parsers."""

from pathlib import Path
from typing import Iterator

import pytest

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import VsdxParser

_THIS_DIR = Path(__file__).parents[3]

_EXAMPLES_DIR = _THIS_DIR / "examples"

# Paths to test VSDX file
FAKE_FILE = _EXAMPLES_DIR / "fake.vsdx"


def _assert_with_parser(parser: BaseBlobParser, splits_by_page: bool = True) -> None:
    """Standard tests to verify that the given parser works.

    Args:
        parser (BaseBlobParser): The parser to test.
        splits_by_page (bool): Whether the parser splits by page or not by default.
    """

    blob = Blob.from_path(FAKE_FILE)
    doc_generator = parser.lazy_parse(blob)
    assert isinstance(doc_generator, Iterator)
    docs = list(doc_generator)

    if splits_by_page:
        assert len(docs) == 14
    else:
        assert len(docs) == 1
    # Test is imprecise since the parsers yield different parse information depending
    # on configuration. Each parser seems to yield a slightly different result
    # for this page!
    assert "This is a title" in docs[0].page_content
    metadata = docs[0].metadata

    assert metadata["source"] == str(FAKE_FILE)

    if splits_by_page:
        assert int(metadata["page"]) == 0


@pytest.mark.requires("xmltodict")
def test_vsdx_parser() -> None:
    """Test the VSDX parser."""
    _assert_with_parser(VsdxParser())
