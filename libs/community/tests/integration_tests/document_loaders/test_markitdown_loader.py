from pathlib import Path

import pytest

from langchain_community.document_loaders import MarkItDownLoader


@pytest.mark.requires("markitdown")
def test_markitdown_loader() -> None:
    """Test MarkItDown loader."""
    file_path = Path(__file__).parent.parent / "examples/example.html"
    loader = MarkItDownLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

    assert docs[0].page_content
