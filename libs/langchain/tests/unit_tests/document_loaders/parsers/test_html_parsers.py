"""Tests for the HTML parsers."""
from pathlib import Path

import pytest

from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers.html import BS4HTMLParser

HERE = Path(__file__).parent
EXAMPLES = HERE.parent.parent.parent / "integration_tests" / "examples"


@pytest.mark.requires("bs4", "lxml")
def test_bs_html_loader() -> None:
    """Test unstructured loader."""
    file_path = EXAMPLES / "example.html"
    blob = Blob.from_path(file_path)
    parser = BS4HTMLParser(get_text_separator="|")
    docs = list(parser.lazy_parse(blob))
    assert isinstance(docs, list)
    assert len(docs) == 1

    metadata = docs[0].metadata
    content = docs[0].page_content

    assert metadata["title"] == "Chew dad's slippers"
    assert metadata["source"] == str(file_path)
    assert content[:2] == "\n|"
