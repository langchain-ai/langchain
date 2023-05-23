from pathlib import Path

from langchain.document_loaders.bibtex import BibtexLoader

BIBTEX_EXAMPLE_FILE = Path(__file__).parent.parent.parent / "examples" / "bibtex.bib"


def test_load_success() -> None:
    """Test that returns one document"""
    loader = BibtexLoader(file_path=str(BIBTEX_EXAMPLE_FILE), load_max_docs=2)
    docs = loader.load()
    assert len(docs) == 1
    print(docs[0].metadata)
    print(docs[0].page_content)

    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        assert set(doc.metadata) == {
            "ID",
            "Published",
            "Title",
            "Publication",
            "Authors",
            "Summary",
            "URL",
        }
