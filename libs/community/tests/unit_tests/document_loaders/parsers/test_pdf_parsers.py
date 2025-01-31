"""Tests for the various PDF parsers."""

import importlib
from pathlib import Path
from typing import Any, Iterator

import pytest

import langchain_community.document_loaders.parsers as pdf_parsers
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import (
    PDFMinerParser,
    PyPDFium2Parser,
    _merge_text_and_extras,
)

_THIS_DIR = Path(__file__).parents[3]

_EXAMPLES_DIR = _THIS_DIR / "examples"

# Paths to test PDF files
HELLO_PDF = _EXAMPLES_DIR / "hello.pdf"
LAYOUT_PARSER_PAPER_PDF = _EXAMPLES_DIR / "layout-parser-paper.pdf"


def test_merge_text_and_extras() -> None:
    assert "abc\n\n\n<image>\n\n<table>\n\n\ndef\n\n\nghi" == _merge_text_and_extras(
        ["<image>", "<table>"], "abc\n\n\ndef\n\n\nghi"
    )
    assert "abc\n\n<image>\n\n<table>\n\ndef\n\nghi" == _merge_text_and_extras(
        ["<image>", "<table>"], "abc\n\ndef\n\nghi"
    )
    assert "abc\ndef\n\n<image>\n\n<table>\n\nghi" == _merge_text_and_extras(
        ["<image>", "<table>"], "abc\ndef\n\nghi"
    )


def _assert_with_parser(parser: BaseBlobParser, *, splits_by_page: bool = True) -> None:
    """Standard tests to verify that the given parser works.

    Args:
        parser (BaseBlobParser): The parser to test.
        splits_by_page (bool): Whether the parser splits by page or not by default.
    """
    blob = Blob.from_path(HELLO_PDF)
    doc_generator = parser.lazy_parse(blob)
    assert isinstance(doc_generator, Iterator)
    docs = list(doc_generator)
    assert len(docs) == 1
    page_content = docs[0].page_content
    assert isinstance(page_content, str)
    # The different parsers return different amount of whitespace, so using
    # startswith instead of equals.
    assert docs[0].page_content.startswith("Hello world!")

    blob = Blob.from_path(LAYOUT_PARSER_PAPER_PDF)
    doc_generator = parser.lazy_parse(blob)
    assert isinstance(doc_generator, Iterator)
    docs = list(doc_generator)

    if splits_by_page:
        assert len(docs) == 16
    else:
        assert len(docs) == 1
    # Test is imprecise since the parsers yield different parse information depending
    # on configuration. Each parser seems to yield a slightly different result
    # for this page!
    assert "LayoutParser" in docs[0].page_content
    metadata = docs[0].metadata

    assert metadata["source"] == str(LAYOUT_PARSER_PAPER_PDF)

    if splits_by_page:
        assert int(metadata["page"]) == 0


@pytest.mark.requires("pdfminer")
def test_pdfminer_parser() -> None:
    """Test PDFMiner parser."""
    # Does not follow defaults to split by page.
    _assert_with_parser(PDFMinerParser(), splits_by_page=False)


@pytest.mark.requires("pypdfium2")
def test_pypdfium2_parser() -> None:
    """Test PyPDFium2 parser."""
    # Does not follow defaults to split by page.
    _assert_with_parser(PyPDFium2Parser())


@pytest.mark.parametrize(
    "parser_factory,require,params",
    [
        ("PyMuPDFParser", "pymupdf", {}),
        ("PyPDFParser", "pypdf", {}),
    ],
)
def test_parsers(
    parser_factory: str,
    require: str,
    params: dict[str, Any],
) -> None:
    try:
        require = require.replace("-", "")
        importlib.import_module(require, package=None)
        parser_class = getattr(pdf_parsers, parser_factory)
        parser = parser_class()
        _assert_with_parser(parser, **params)
    except ModuleNotFoundError:
        pytest.skip(f"{parser_factory} skiped. Require '{require}'")
