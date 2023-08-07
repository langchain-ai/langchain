"""Tests for the various PDF parsers."""
from pathlib import Path
from typing import Iterator

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers.pdf import (
    PDFMinerParser,
    PDFPlumberParser,
    PyMuPDFParser,
    PyPDFium2Parser,
    PyPDFParser,
)

# PDFs to test parsers on.
HELLO_PDF = Path(__file__).parent.parent.parent / "examples" / "hello.pdf"

LAYOUT_PARSER_PAPER_PDF = (
    Path(__file__).parent.parent.parent / "examples" / "layout-parser-paper.pdf"
)


def _assert_with_parser(parser: BaseBlobParser, splits_by_page: bool = True) -> None:
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
        assert metadata["page"] == 0


def test_pymupdf_loader() -> None:
    """Test PyMuPDF loader."""
    _assert_with_parser(PyMuPDFParser())


def test_pypdf_parser() -> None:
    """Test PyPDF parser."""
    _assert_with_parser(PyPDFParser())


def test_pdfminer_parser() -> None:
    """Test PDFMiner parser."""
    # Does not follow defaults to split by page.
    _assert_with_parser(PDFMinerParser(), splits_by_page=False)


def test_pypdfium2_parser() -> None:
    """Test PyPDFium2 parser."""
    # Does not follow defaults to split by page.
    _assert_with_parser(PyPDFium2Parser())


def test_pdfplumber_parser() -> None:
    """Test PDFPlumber parser."""
    _assert_with_parser(PDFPlumberParser())
