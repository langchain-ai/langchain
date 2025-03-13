"""Tests for the various PDF parsers."""

import importlib
from pathlib import Path
from typing import Any, Iterator, Type

import pytest

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import (
    PDFMinerParser,
    PDFPlumberParser,
    PyMuPDFParser,
    PyPDFium2Parser,
    PyPDFParser,
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


@pytest.mark.parametrize(
    "parser_class,require,ctr_params,params",
    [
        (PDFMinerParser, "pdfminer", {}, {"splits_by_page": False}),
        (PDFPlumberParser, "pdfplumber", {"metadata_format": "standard"}, {}),
        (PyMuPDFParser, "pymupdf", {}, {}),
        (PyPDFParser, "pypdf", {}, {}),
        (PyPDFium2Parser, "pypdfium2", {}, {}),
    ],
)
def test_parsers(
    parser_class: Type,
    require: str,
    ctr_params: dict[str, Any],
    params: dict[str, Any],
) -> None:
    try:
        require = require.replace("-", "")
        importlib.import_module(require, package=None)
        parser = parser_class(**ctr_params)
        _assert_with_parser(parser, **params)
    except ModuleNotFoundError:
        pytest.skip(f"{parser_class} skiped. Require '{require}'")
