"""Tests for the various PDF parsers."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pytest

import langchain_community.document_loaders.parsers as pdf_parsers
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import (
    BaseImageBlobParser,
    PDFPlumberParser,
)

if TYPE_CHECKING:
    from PIL.Image import Image


# PDFs to test parsers on.
HELLO_PDF = Path(__file__).parent.parent.parent / "examples" / "hello.pdf"

LAYOUT_PARSER_PAPER_PDF = (
    Path(__file__).parent.parent.parent / "examples" / "layout-parser-paper.pdf"
)

LAYOUT_PARSER_PAPER_PASSWORD_PDF = (
    Path(__file__).parent.parent.parent
    / "examples"
    / "layout-parser-paper-password.pdf"
)

DUPLICATE_CHARS = (
    Path(__file__).parent.parent.parent / "examples" / "duplicate-chars.pdf"
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
    assert re.findall(r"Hello\s+world!", docs[0].page_content)

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


def _assert_with_duplicate_parser(parser: BaseBlobParser, dedupe: bool = False) -> None:
    """PDFPlumber tests to verify that duplicate characters appear or not
    Args:
        parser (BaseBlobParser): The parser to test.
        splits_by_page (bool): Whether the parser splits by page or not by default.
        dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
    """
    blob = Blob.from_path(DUPLICATE_CHARS)
    doc_generator = parser.lazy_parse(blob)
    assert isinstance(doc_generator, Iterator)
    docs = list(doc_generator)

    if dedupe:
        # use dedupe avoid duplicate characters.
        assert "1000 Series" == docs[0].page_content.split("\n")[0]
    else:
        # duplicate characters will appear in doc if not dedupe
        assert "11000000 SSeerriieess" == docs[0].page_content.split("\n")[0]


def test_pdfplumber_parser() -> None:
    """Test PDFPlumber parser."""
    _assert_with_parser(PDFPlumberParser())
    _assert_with_duplicate_parser(PDFPlumberParser())
    _assert_with_duplicate_parser(PDFPlumberParser(dedupe=True), dedupe=True)


class EmptyImageBlobParser(BaseImageBlobParser):
    def _analyze_image(self, img: "Image") -> str:
        return "Hello world"


@pytest.mark.parametrize(
    "mode,image_parser",
    [("single", EmptyImageBlobParser()), ("page", None)],
)
@pytest.mark.parametrize(
    "parser_factory,params",
    [
        ("PDFMinerParser", {}),
        ("PyMuPDFParser", {}),
        ("PyPDFium2Parser", {}),
        ("PyPDFParser", {"extraction_mode": "plain"}),
        ("PyPDFParser", {"extraction_mode": "layout"}),
    ],
)
@pytest.mark.requires("pillow")
def test_mode_and_extract_images_variations(
    parser_factory: str,
    params: dict,
    mode: str,
    image_parser: BaseImageBlobParser,
) -> None:
    _test_matrix(
        parser_factory,
        params,
        mode,
        image_parser,
        images_inner_format="text",
    )


@pytest.mark.parametrize(
    "images_inner_format",
    ["text", "markdown-img", "html-img"],
)
@pytest.mark.parametrize(
    "parser_factory,params",
    [
        ("PDFMinerParser", {}),
        ("PyMuPDFParser", {}),
        ("PyPDFium2Parser", {}),
        ("PyPDFParser", {"extraction_mode": "plain"}),
        ("PyPDFParser", {"extraction_mode": "layout"}),
    ],
)
@pytest.mark.requires("pillow")
def test_mode_and_image_formats_variations(
    parser_factory: str,
    params: dict,
    images_inner_format: str,
) -> None:
    mode = "single"
    image_parser = EmptyImageBlobParser()

    _test_matrix(
        parser_factory,
        params,
        mode,
        image_parser,
        images_inner_format,
    )


def _test_matrix(
    parser_factory: str,
    params: dict,
    mode: str,
    image_parser: BaseImageBlobParser,
    images_inner_format: str,
) -> None:
    """Apply the same test for all *standard* PDF parsers.

    - Try with mode `single` and `page`
    - Try with image_parser `None` or others
    """

    def _std_assert_with_parser(parser: BaseBlobParser) -> None:
        """Standard tests to verify that the given parser works.

        Args:
            parser (BaseBlobParser): The parser to test.
        """
        blob = Blob.from_path(LAYOUT_PARSER_PAPER_PDF)
        doc_generator = parser.lazy_parse(blob)
        docs = list(doc_generator)
        metadata = docs[0].metadata
        assert metadata["source"] == str(LAYOUT_PARSER_PAPER_PDF)
        assert "creationdate" in metadata
        assert "creator" in metadata
        assert "producer" in metadata
        assert "total_pages" in metadata
        if len(docs) > 1:
            assert metadata["page"] == 0
        if hasattr(parser, "extract_images") and parser.extract_images:
            images = []
            for doc in docs:
                _HTML_image = (
                    r"<img\s+[^>]*"
                    r'src="([^"]+)"(?:\s+alt="([^"]*)")?(?:\s+'
                    r'title="([^"]*)")?[^>]*>'
                )
                _markdown_image = r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"([^\"]+)\")?\)"
                match = re.findall(_markdown_image, doc.page_content)
                if match:
                    images.extend(match)
            assert len(images) >= 1

        if hasattr(parser, "password"):
            old_password = parser.password
            parser.password = "password"
            blob = Blob.from_path(LAYOUT_PARSER_PAPER_PASSWORD_PDF)
            doc_generator = parser.lazy_parse(blob)
            docs = list(doc_generator)
            assert len(docs)
            parser.password = old_password

    parser_class = getattr(pdf_parsers, parser_factory)

    parser = parser_class(
        mode=mode,
        images_parser=image_parser,
        images_inner_format=images_inner_format,
        **params,
    )
    _assert_with_parser(parser, splits_by_page=(mode == "page"))
    _std_assert_with_parser(parser)


@pytest.mark.parametrize(
    "mode",
    ["single", "page"],
)
@pytest.mark.parametrize(
    "extract_tables",
    ["markdown", "html", "csv", None],
)
@pytest.mark.parametrize(
    "parser_factory,params",
    [
        ("PyMuPDFParser", {}),
    ],
)
def test_parser_with_table(
    parser_factory: str,
    params: dict,
    mode: str,
    extract_tables: str,
) -> None:
    from PIL.Image import Image

    from langchain_community.document_loaders.parsers.images import BaseImageBlobParser

    def _std_assert_with_parser(parser: BaseBlobParser) -> None:
        """Standard tests to verify that the given parser works.

        Args:
            parser (BaseBlobParser): The parser to test.
        """
        blob = Blob.from_path(LAYOUT_PARSER_PAPER_PDF)
        doc_generator = parser.lazy_parse(blob)
        docs = list(doc_generator)
        tables = []
        for doc in docs:
            if extract_tables == "markdown":
                pattern = (
                    r"(?s)("
                    r"(?:(?:[^\n]*\|)\n)"
                    r"(?:\|(?:\s?:?---*:?\s?\|)+)\n"
                    r"(?:(?:[^\n]*\|)\n)+"
                    r")"
                )
            elif extract_tables == "html":
                pattern = r"(?s)(<table[^>]*>(?:.*?)<\/table>)"
            elif extract_tables == "csv":
                pattern = (
                    r"((?:(?:"
                    r'(?:"(?:[^"]*(?:""[^"]*)*)"'
                    r"|[^\n,]*),){2,}"
                    r"(?:"
                    r'(?:"(?:[^"]*(?:""[^"]*)*)"'
                    r"|[^\n]*))\n){2,})"
                )
            else:
                pattern = None
            if pattern:
                matches = re.findall(pattern, doc.page_content)
                if matches:
                    tables.extend(matches)
        if extract_tables:
            assert len(tables) >= 1
        else:
            assert not len(tables)

    class EmptyImageBlobParser(BaseImageBlobParser):
        def _analyze_image(self, img: Image) -> str:
            return "![image](.)"

    parser_class = getattr(pdf_parsers, parser_factory)

    parser = parser_class(
        mode=mode,
        extract_tables=extract_tables,
        images_parser=EmptyImageBlobParser(),
        **params,
    )
    _std_assert_with_parser(parser)
