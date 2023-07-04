from pathlib import Path

from langchain.document_loaders import JSONLinesLoader


def test_jsonl_loader() -> None:
    """Test unstructured loader."""
    file_path = Path(__file__).parent.parent / "examples/example.jsonl"
    loader = JSONLinesLoader(str(file_path), ".messages[].content")
    docs = loader.load()

    # Check that the correct number of documents are loaded.
    assert len(docs) == 6
