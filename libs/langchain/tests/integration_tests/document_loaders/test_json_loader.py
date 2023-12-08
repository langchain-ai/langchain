from pathlib import Path

from langchain.document_loaders import JSONLoader


def test_json_loader() -> None:
    """Test unstructured loader."""
    file_path = Path(__file__).parent.parent / "examples/example.json"
    loader = JSONLoader(str(file_path), ".messages[].content")
    docs = loader.load()

    # Check that the correct number of documents are loaded.
    assert len(docs) == 3

    # Make sure that None content are converted to empty strings.
    assert docs[-1].page_content == ""
