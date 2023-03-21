from pathlib import Path

from langchain.document_loaders.html_bs import BSHTMLLoader


def test_bs_html_loader() -> None:
    """Test unstructured loader."""
    file_path = Path(__file__).parent.parent / "examples/example.html"
    loader = BSHTMLLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

    metadata = docs[0].metadata

    assert metadata["title"] == "Chew dad's slippers"
    assert metadata["source"] == str(file_path)
