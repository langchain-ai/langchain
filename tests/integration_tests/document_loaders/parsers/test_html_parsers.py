"""Tests for the HTML parsers."""
from pathlib import Path
from typing import Iterator

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers.html import BS4HTMLParser

# PDFs to test parsers on.
HELLO_PDF = Path(__file__).parent.parent.parent / "examples" / "hello.pdf"

LAYOUT_PARSER_PAPER_PDF = (
    Path(__file__).parent.parent.parent / "examples" / "layout-parser-paper.pdf"
)
