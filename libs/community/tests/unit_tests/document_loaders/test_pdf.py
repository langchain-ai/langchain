import re
from pathlib import Path

import pytest

from langchain_community.document_loaders import PyPDFLoader

path_to_simple_pdf = (
    Path(__file__).parent.parent.parent / "integration_tests/examples/hello.pdf"
)
path_to_layout_pdf = (
    Path(__file__).parent.parent
    / "document_loaders/sample_documents/layout-parser-paper.pdf"
)
path_to_layout_pdf_txt = (
    Path(__file__).parent.parent.parent
    / "integration_tests/examples/layout-parser-paper-page-1.txt"
)


@pytest.mark.requires("pypdf")
def test_pypdf_loader() -> None:
    """Test PyPDFLoader."""
    loader = PyPDFLoader(path_to_simple_pdf)
    docs = loader.load()

    assert len(docs) == 1

    loader = PyPDFLoader(path_to_layout_pdf)

    docs = loader.load()
    assert len(docs) == 16
    for page, doc in enumerate(docs):
        assert doc.metadata["page"] == page
        assert doc.metadata["source"].endswith("layout-parser-paper.pdf")
        assert len(doc.page_content) > 10

    first_page = docs[0].page_content
    for expected in ["LayoutParser", "A Uniﬁed Toolkit"]:
        assert expected in first_page


@pytest.mark.requires("pypdf")
def test_pypdf_loader_with_layout() -> None:
    """Test PyPDFLoader with layout mode."""
    loader = PyPDFLoader(path_to_layout_pdf, extraction_mode="layout")

    docs = loader.load()
    assert len(docs) == 16
    for page, doc in enumerate(docs):
        assert doc.metadata["page"] == page
        assert doc.metadata["source"].endswith("layout-parser-paper.pdf")
        assert len(doc.page_content) > 10

    first_page = docs[0].page_content
    for expected in ["LayoutParser", "A Uniﬁed Toolkit"]:
        assert expected in first_page

    expected = path_to_layout_pdf_txt.read_text(encoding="utf-8")
    cleaned_first_page = re.sub(r"\x00", "", first_page)
    cleaned_expected = re.sub(r"\x00", "", expected)
    assert cleaned_first_page == cleaned_expected.strip()
