from pathlib import Path

import pytest

from langchain.document_loaders.bibtex import BibtexLoader

BIBTEX_EXAMPLE_FILE = Path(__file__).parent / "sample_documents" / "bibtex.bib"


@pytest.mark.requires("fitz", "bibtexparser")
def test_load_success() -> None:
    """Test that returns one document"""
    loader = BibtexLoader(file_path=str(BIBTEX_EXAMPLE_FILE))
    docs = loader.load()
    assert len(docs) == 1
    doc = docs[0]
    assert doc.page_content
    assert set(doc.metadata) == {
        "id",
        "published_year",
        "title",
        "publication",
        "authors",
        "abstract",
    }


@pytest.mark.requires("fitz", "bibtexparser")
def test_load_max_content_chars() -> None:
    """Test that cuts off document contents at max_content_chars."""
    loader = BibtexLoader(file_path=str(BIBTEX_EXAMPLE_FILE), max_content_chars=10)
    doc = loader.load()[0]
    assert len(doc.page_content) == 10


@pytest.mark.requires("fitz", "bibtexparser")
def test_load_load_extra_metadata() -> None:
    """Test that returns extra metadata fields."""
    loader = BibtexLoader(file_path=str(BIBTEX_EXAMPLE_FILE), load_extra_metadata=True)
    doc = loader.load()[0]
    assert set(doc.metadata) == {
        "id",
        "published_year",
        "title",
        "publication",
        "authors",
        "abstract",
        "booktitle",
        "editor",
        "organization",
    }


@pytest.mark.requires("fitz", "bibtexparser")
def test_load_file_pattern() -> None:
    """Test that returns no documents when json file pattern specified."""
    loader = BibtexLoader(
        file_path=str(BIBTEX_EXAMPLE_FILE), file_pattern=r"[^:]+\.json"
    )
    docs = loader.load()
    assert len(docs) == 0
