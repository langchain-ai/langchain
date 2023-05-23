from pathlib import Path

import pytest

from langchain.document_loaders.bibtex import BibtexLoader

BIBTEX_EXAMPLE_FILE = (
    Path(__file__).parent.parent.parent
    / "integration_tests"
    / "examples"
    / "bibtex.bib"
)


@pytest.mark.requires("pymupdf")
def test_load_success() -> None:
    """Test that returns one document"""
    loader = BibtexLoader(file_path=str(BIBTEX_EXAMPLE_FILE), load_max_docs=2)
    docs = loader.load()
    assert len(docs) == 1

    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        assert set(doc.metadata) == {
            "id",
            "published",
            "title",
            "publication",
            "authors",
            "summary",
            "url",
            "editor",
            "booktitle",
            "organization",
        }
